module KalmanFilter

    using DocStringExtensions, JuliennedArrays, Distributions, LinearAlgebra, LazyArrays

    export
        WanMerveWeightingParameters,
        MeanSetWeightingParameters,
        GaussSetWeightingParameters,
        ScaledSetWeightingParameters,
        Augment,
        KalmanInits,
        KFTUIntermediate,
        KFMUIntermediate,
        state,
        covariance,
        innovation,
        innovation_covariance,
        kalman_gain,
        time_update,
        time_update!,
        measurement_update,
        measurement_update!

    struct KalmanInits{X,P}
        state::X
        covariance::P
    end

    abstract type AbstractTimeUpdate end
    abstract type AbstractMeasurementUpdate end

    state(kf::Union{<: AbstractTimeUpdate, <: AbstractMeasurementUpdate, <: KalmanInits}) = kf.state
    covariance(kf::Union{<: AbstractTimeUpdate, <: AbstractMeasurementUpdate, <: KalmanInits}) = kf.covariance
    innovation(kf::AbstractMeasurementUpdate) = kf.innovation
    innovation_covariance(kf::AbstractMeasurementUpdate) = kf.innovation_covariance
    kalman_gain(kf::AbstractMeasurementUpdate) = kf.kalman_gain

    include("kf.jl")
    include("sigmapoints.jl")
    include("aukf.jl")
    include("ukf.jl")
    include("tests.jl")

end
