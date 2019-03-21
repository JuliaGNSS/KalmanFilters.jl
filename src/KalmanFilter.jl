module KalmanFilter

    using DocStringExtensions, Distributions, LinearAlgebra, LazyArrays, Statistics
    import Statistics: mean, cov

    export
        WanMerweWeightingParameters,
        MeanSetWeightingParameters,
        GaussSetWeightingParameters,
        ScaledSetWeightingParameters,
        Augment,
        KFTUIntermediate,
        UKFTUIntermediate,
        KFMUIntermediate,
        UKFMUIntermediate,
        state,
        covariance,
        innovation,
        innovation_covariance,
        kalman_gain,
        time_update,
        time_update!,
        measurement_update,
        measurement_update!,
        nis,
        nis_test,
        sigma_bound_test,
        two_sigma_bound_test

    abstract type AbstractTimeUpdate end
    abstract type AbstractMeasurementUpdate end
    abstract type AbstractWeightingParameters end

    state(kf::Union{<: AbstractTimeUpdate, <: AbstractMeasurementUpdate}) = kf.state
    covariance(kf::Union{<: AbstractTimeUpdate, <: AbstractMeasurementUpdate}) = kf.covariance
    innovation(kf::AbstractMeasurementUpdate) = kf.innovation
    innovation_covariance(kf::AbstractMeasurementUpdate) = kf.innovation_covariance
    kalman_gain(kf::AbstractMeasurementUpdate) = kf.kalman_gain

    @static if VERSION < v"1.1"
        eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))
    end

    include("kf.jl")
    include("sigmapoints.jl")
    include("ukf.jl")
    include("aukf.jl")
    include("tests.jl")

end
