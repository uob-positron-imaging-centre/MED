# File   : view_gp.jl
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 04.04.2022


using TOML
using DelimitedFiles

using Printf
using LinearAlgebra
using Statistics

using GLMakie
using Colors




# This GP implementation follows the one used in Marcus Noack's fvGP library,
# with some aggressive optimisation and caching for interactive plotting.
# Thank you.
function matern(d)
    (1. + sqrt(3.) * d) * exp(-sqrt(3.) * d)
end


function kernel(x1, x2, hp)
    distances = zeros(size(x1, 1), size(x2, 1))

    for j in 1:size(x2, 1)
        for i in 1:size(x1, 1)
            for k in 1:length(hp) - 1
                distances[i, j] += ((x2[j, k] - x1[i, k]) / hp[1 + k]) ^ 2
            end

            distances[i, j] = hp[1] * matern(sqrt(distances[i, j]))
        end
    end

    distances
end


function cov_prod(x, y, hyperparameters)
    k = kernel(x, x, hyperparameters)

    ys = y .- mean(y)
    xs = k \ ys

    xs
end


function predict(ymean, ktp, cov_train)
    ktp' * cov_train .+ ymean
end


function variance(ktp, kpp, kinv)
    # Efficiently compute diag(kpp - ktp' * kinv * ktp)
    temp = ktp' * kinv
    v = diag(kpp)

    for i in 1:size(temp, 1)
        for j in 1:size(ktp, 1)
            v[i] -= temp[i, j] * ktp[j, i]
        end
    end

    v
end


function meshgrid(x::AbstractVector, y::AbstractVector)
    xx = zeros(length(x), length(y))
    yy = zeros(length(x), length(y))

    for j in 1:length(y)
        for i in 1:length(x)
            xx[i, j] = x[i]
            yy[i, j] = y[j]
        end
    end

    xx, yy
end


function generate_plot!(med, fig, sliders, resolution)
    # Cache computations
    cmap = :Blues_9 |> Reverse |> to_colormap
    d = length(med.parameters)
    ymean = mean(med.ytrain)
    cov_train = cov_prod(med.xtrain, med.ytrain, med.hyperparameters)

    ktt = kernel(med.xtrain, med.xtrain, med.hyperparameters)
    ktt_inv = inv(ktt)

    # Pre-allocate arrays
    xpred = zeros(resolution[1] * resolution[2], d)     # Points to predict GP 2D slices at
    cond = BitVector(undef, size(med.xtrain, 1))        # Conditional to plot points
    msize = zeros(size(med.xtrain, 1))                  # Marker size, proportional to distance

    sresolution = 200
    xspred = zeros(sresolution, d)                      # Points to predict GP single lines at

    # Fixed parameters for the remaining factors beside the given 2D slice
    intervals = med.maxs .- med.mins

    # Maximum marker size
    maxsize = 15.

    # Plot GP-predicted response and uncertainty for each pair of parameters, while holding
    # everything else constant, as set by the sliders
    for i in 1:length(med.parameters) - 1
        for j in i + 1:length(med.parameters)

            # Generate values from a flattened grid
            x = LinRange(med.mins[i], med.maxs[i], resolution[1])
            y = LinRange(med.mins[j], med.maxs[j], resolution[2])
            xx, yy = meshgrid(x, y)

            # Create Observable: for the given fixed parameter values (from the sliders) compute
            # the GP predictions, variances and samples to be plotted
            predicted = lift([sl.value for sl in sliders]...) do fixed...

                # Create flattened 2D slices to evaluate the GP at; fix the other parameters
                xpred .= 0.
                xpred[:, i] .= @view xx[:]
                xpred[:, j] .= @view yy[:]

                # Only plot evaluated points that are close to the slice and set their markersize
                # to be proportional to how close they are to the plane
                cond .= true
                maxdist = 0.
                msize .= 0.

                d = length(med.parameters)
                for k in 1:d
                    if k != i && k != j
                        xpred[:, k] .= fixed[k]

                        cond .&= (fixed[k] - 0.5fixed[k + d] * intervals[k]) .< (@view med.xtrain[:, k])
                        cond .&= (fixed[k] + 0.5fixed[k + d] * intervals[k]) .> (@view med.xtrain[:, k])

                        maxdist += (0.5fixed[k + d] * intervals[k]) ^ 2
                        msize .+= ((@view med.xtrain[:, k]) .- fixed[k]) .^ 2
                    end
                end

                # Set marker size to be inversely proportional to how close it is to slice
                if d != 2
                    maxdist = sqrt(maxdist)
                    msize .= maxsize .- (maxsize / maxdist) .* sqrt.(msize)
                else
                    msize .= maxsize / 2
                end

                # Pre-compute kernels
                ktp = kernel(med.xtrain, xpred, med.hyperparameters)
                kpp = kernel(xpred, xpred, med.hyperparameters)

                # Evaluate GP at given parameter values and reshape to 2D slice
                ypred = reshape(
                    predict(ymean, ktp, cov_train),
                    resolution[1],
                    resolution[2],
                )

                # Evaluate GP variance / uncertainty
                yvar = reshape(
                    variance(ktp, kpp, ktt_inv),
                    resolution[1],
                    resolution[2],
                )

                # Close points sampled
                samples = [(@view med.xtrain[cond, i])'; (@view med.xtrain[cond, j])']


                ypred, yvar, samples, msize[cond]
            end

            # "Extract" Observables by tying a new Observable to each Tuple element to be plotted,
            # so that every time the $predicted inputs change, so will the extracted Observables
            ypred = @lift($predicted[1])
            yvar = @lift($predicted[2])
            samples = @lift($predicted[3])
            markersize = @lift($predicted[4])

            # Plot predictions
            ax1 = Axis(fig[j, 2 * i - 1]; xlabel = med.parameters[i], ylabel = med.parameters[j])

            hmap_pred = contourf!(ax1, x, y, ypred; colormap = cmap)
            Colorbar(fig[j, 2 * i], hmap_pred, #=label = "Response"=#)

            scatter_pred = scatter!(
                ax1,
                samples,
                markersize = markersize,
                # marker = :xcross,
                color = colorant"rgb(8,48,107)",
                strokewidth = 1,
                strokecolor = :white,
            )

            # Plot uncertainties
            ax2 = Axis(fig[i, 2 * j - 1]; xlabel = med.parameters[i], ylabel = med.parameters[j])

            hmap_var  = contourf!(ax2, x, y, yvar; colormap = cmap)
            Colorbar(fig[i, 2 * j], hmap_var, #=label = "Variance"=#)

            scatter_pred = scatter!(
                ax2,
                samples,
                markersize = markersize,
                # marker = :xcross,
                color = colorant"rgb(8,48,107)",
                strokewidth = 1,
                strokecolor = :white,
            )
        end
    end

    # Plot GP-predicted response and uncertainty for each parameter while holding everything
    # else constant, as set by sliders
    for i in 1:length(med.parameters)

        sx = LinRange(med.mins[i], med.maxs[i], sresolution) |> Vector

        # Create Observable: for the given fixed parameter values (from the sliders) compute
        # the GP predictions, variances and samples to be plotted
        spredicted = lift([sl.value for sl in sliders]...) do fixed...

            # Create 1D lines of values to evaluate the GP at; fix the other parameters
            xspred .= 0.
            xspred[:, i] .= sx

            # Only plot evaluated points that are close to the slice and set their markersize
            # to be proportional to how close they are to the plane
            cond .= true
            maxdist = 0.
            msize .= 0.

            d = length(med.parameters)
            for k in 1:d
                if k != i
                    xspred[:, k] .= fixed[k]

                    cond .&= (fixed[k] - 0.5fixed[k + d] * intervals[k]) .< (@view med.xtrain[:, k])
                    cond .&= (fixed[k] + 0.5fixed[k + d] * intervals[k]) .> (@view med.xtrain[:, k])

                    maxdist += (0.5fixed[k + d] * intervals[k]) ^ 2
                    msize .+= ((@view med.xtrain[:, k]) .- fixed[k]) .^ 2
                end
            end

            # Set marker size to be inversely proportional to how close it is to slice
            maxdist = sqrt(maxdist)
            msize .= maxsize .- (maxsize / maxdist) .* sqrt.(msize)

            # Pre-compute kernels
            ktp = kernel(med.xtrain, xspred, med.hyperparameters)
            kpp = kernel(xspred, xspred, med.hyperparameters)

            # Evaluate GP at given parameter values and reshape to 2D slice
            ypred = predict(ymean, ktp, cov_train)

            # Evaluate GP variance / uncertainty
            yvar = variance(ktp, kpp, ktt_inv)

            # Close points sampled
            samples = [med.xtrain[cond, i]'; med.ytrain[cond]']

            ypred, ypred - yvar, ypred + yvar, samples, msize[cond]
        end

        # "Extract" Observables by tying a new Observable to each Tuple element to be plotted,
        # so that every time the $predicted inputs change, so will the extracted Observables
        sypred = @lift($spredicted[1])
        syvar_lo = @lift($spredicted[2])
        syvar_hi = @lift($spredicted[3])
        ssamples = @lift($spredicted[4])
        smarkersize = @lift($spredicted[5])

        # Plot predictions
        ax = Axis(fig[i, 2 * i - 1:2 * i]; xlabel = med.parameters[i], #=ylabel = "Response"=#)

        band!(ax, sx, syvar_lo, syvar_hi, color = colorant"rgb(107,174,214)")
        lines!(ax, sx, sypred, color = colorant"rgb(8,81,156)")

        scatter!(
            ax,
            ssamples,
            markersize = smarkersize,
            # marker = :xcross,
            color = colorant"rgb(8,48,107)",
            strokewidth = 1,
            strokecolor = :white,
        )

        on(spredicted) do _
            reset_limits!(ax)
        end
    end

    fig
end


@inbounds @fastmath function generate_plot(med, resolution=(64, 64))
    # Create figure and sliders
    fig = Figure(resolution = (1400, 1000))

    # Formatting parameter labels
    formatter(param, val) = @sprintf "%s = %4.4f" param val

    # Create sliders and labels showing the current slider value
    d = length(med.parameters)
    sliders = Vector{Any}(undef, 2d)
    for i in 1:d
        sliders[i] = Slider(
            fig[d + 1, 2i - 1:2i],
            range = LinRange(med.mins[i], med.maxs[i], 50),
            startvalue = 0.5 * (med.mins[i] + med.maxs[i]),
            snap = false,
            color_active = colorant"rgb(8,48,107)",
            color_inactive = colorant"rgb(222,235,247)",
            color_active_dimmed = colorant"rgb(33,113,181)"
        )

        label = Label(
            fig[d + 2, 2i - 1:2i],
            formatter(med.parameters[i], sliders[i].value[]),
        )

        on(sliders[i].value) do value
            label.text = formatter(med.parameters[i], value)
        end
    end

    # Create sliders for slice widths
    for i in 1:d
        sliders[d + i] = Slider(
            fig[d + 3, 2i - 1:2i],
            range = 0:0.05:1,
            startvalue = 0.3,
            snap = false,
            color_active = colorant"rgb(8,48,107)",
            color_inactive = colorant"rgb(222,235,247)",
            color_active_dimmed = colorant"rgb(33,113,181)"
        )

        label = Label(
            fig[d + 4, 2i - 1:2i],
            formatter("Slice Width", sliders[d + i].value[]),
        )

        on(sliders[d + i].value) do value
            label.text = formatter("Slice Width", value)
        end
    end

    generate_plot!(med, fig, sliders |> Tuple, resolution |> Tuple)
end


function read_med(medpath, response_index)
    setup = TOML.parsefile("$medpath/setup.toml")
    data = readdlm("$medpath/results.csv", ',', skipstart=1)

    mins = setup["parameters"]["min"]
    maxs = setup["parameters"]["max"]
    parameters = setup["parameter_names"]

    # Data to train Gaussian Processes on
    xtrain = data[:, 1:length(parameters)] |> Matrix
    ytrain = data[:, length(parameters) + 2 * response_index] |> Vector

    # Save training data in a NamedTuple with Tuples to maximise optimisations
    med = (
        parameters = parameters |> Tuple,
        xtrain = xtrain,
        ytrain = ytrain,
        hyperparameters = setup["hyperparameters"][response_index] |> Tuple,
        mins = [mins[p] for p in parameters] |> Tuple,
        maxs = [maxs[p] for p in parameters] |> Tuple,
    )

    med
end


"""
    plot_gp(medpath[, response_index=1, resolution=(32, 32)])

Plot a system response & uncertainty saved by a MED object at `medpath`.

Plots 2D slices through the parameter space of the Gaussian Process quantifying outputs and
variances of a single response of interest (eg `response_index = 1` returns the first
response).

# Examples

```julia-repl
julia> include("plot_gp.jl")
julia> plot_gp("med_seed123") |> display
```
"""
function plot_gp(medpath, response_index=1, resolution=(32, 32))
    med = read_med(medpath, response_index)
    fig = generate_plot(med, resolution)

    fig
end


# If run from the command-line, use cmdargs. Call this script as:
# $> julia plot_gp.jl med_save_path response_index resolution1 resolution2
if abspath(PROGRAM_FILE) == @__FILE__
    medpath = ARGS[1]
    response_index = parse(Int64, ARGS[2])
    resolution = (parse(Int64, ARGS[3]), parse(Int64, ARGS[4]))

    fig = plot_gp(medpath, response_index, resolution)
    display(fig)

    # Keep Julia running while GLMakie window is open
    while events(fig).window_open[]
        sleep(0.1)
    end
end
