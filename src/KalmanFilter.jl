module KalmanFilter

    using
        DocStringExtensions,
        Distributions,
        LinearAlgebra,
        LazyArrays,
        Statistics

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
        get_state,
        get_covariance,
        get_innovation,
        get_innovation_covariance,
        get_kalman_gain,
        get_sqrt_covariance,
        get_sqrt_innovation_covariance,
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
    abstract type AbstractSRTimeUpdate <: AbstractTimeUpdate end
    abstract type AbstractSRMeasurementUpdate <: AbstractMeasurementUpdate end
    abstract type AbstractWeightingParameters end

    function get_state(kf::Union{<: AbstractTimeUpdate, <: AbstractMeasurementUpdate})
        kf.state
    end
    function get_covariance(
        kf::Union{<: AbstractTimeUpdate, <: AbstractMeasurementUpdate}
    )
        kf.covariance
    end
    get_innovation(kf::AbstractMeasurementUpdate) = kf.innovation
    get_innovation_covariance(kf::AbstractMeasurementUpdate) = kf.innovation_covariance
    get_kalman_gain(kf::AbstractMeasurementUpdate) = kf.kalman_gain

    function get_covariance(
        kf::Union{<: AbstractSRTimeUpdate, <: AbstractSRMeasurementUpdate}
    )
        kf.covariance.L * kf.covariance.U
    end

    function get_sqrt_covariance(
        kf::Union{<: AbstractSRTimeUpdate, <: AbstractSRMeasurementUpdate}
    )
        kf.covariance
    end

    function get_innovation_covariance(
        kf::Union{<: AbstractSRTimeUpdate, <: AbstractSRMeasurementUpdate}
    )
        kf.innovation_covariance.L * kf.innovation_covariance.U
    end

    function get_sqrt_innovation_covariance(
        kf::Union{<: AbstractSRTimeUpdate, <: AbstractSRMeasurementUpdate}
    )
        kf.innovation_covariance
    end

    @static if VERSION < v"1.1"
        eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))
    end

    include("kf.jl")
    include("srkf.jl")
    include("sigmapoints.jl")
    include("ukf.jl")
    include("aukf.jl")
    include("srukf.jl")
    include("sraukf.jl")
    include("tests.jl")

end
