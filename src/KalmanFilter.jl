module KalmanFilter

    using
        DocStringExtensions,
        Distributions,
        LinearAlgebra,
        LazyArrays,
        Statistics,
        InvertedIndices

    const liblapack = Base.liblapack_name
    
    import Statistics: mean, cov

    import ..LinearAlgebra.BLAS.@blasfunc

    import ..LinearAlgebra: BlasFloat, BlasInt,
        DimensionMismatch, chkstride1, checksquare, cholesky

    export
        WanMerweWeightingParameters,
        MeanSetWeightingParameters,
        GaussSetWeightingParameters,
        ScaledSetWeightingParameters,
        Augment,
        KFTUIntermediate,
        SRKFTUIntermediate,
        SRUKFTUIntermediate,
        SRAUKFTUIntermediate,
        AUKFTUIntermediate,
        UKFTUIntermediate,
        KFMUIntermediate,
        SRKFMUIntermediate,
        UKFMUIntermediate,
        SRUKFMUIntermediate,
        SRAUKFMUIntermediate,
        AUKFMUIntermediate,
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
        calc_nis,
        nis_test,
        sigma_bound_test,
        two_sigma_bound_test,
        ConsideredState,
        ConsideredCovariance,
        ConsideredMeasurementModel,
        ConsideredProcessModel

    abstract type AbstractUpdate{X,P} end
    abstract type AbstractTimeUpdate{X,P} <: AbstractUpdate{X,P} end
    abstract type AbstractMeasurementUpdate{X,P} <: AbstractUpdate{X,P} end
    abstract type AbstractWeightingParameters end

    function get_state(kf::AbstractUpdate)
        kf.state
    end
    function get_covariance(kf::AbstractUpdate)
        kf.covariance
    end
    get_innovation(kf::AbstractMeasurementUpdate) = kf.innovation
    get_innovation_covariance(kf::AbstractMeasurementUpdate) = kf.innovation_covariance
    get_kalman_gain(kf::AbstractMeasurementUpdate) = kf.kalman_gain

    function get_covariance(kf::AbstractUpdate{X,P}) where {X,P <: Cholesky}
        kf.covariance.L * kf.covariance.U
    end

    function get_sqrt_covariance(kf::AbstractUpdate{X,P}) where {X,P <: Cholesky}
        kf.covariance
    end

    function get_innovation_covariance(
        kf::AbstractMeasurementUpdate{X,P}
    ) where {X,P <: Cholesky}
        kf.innovation_covariance.L * kf.innovation_covariance.U
    end

    function get_sqrt_innovation_covariance(
        kf::AbstractMeasurementUpdate{X,P}
    ) where {X,P <: Cholesky}
        kf.innovation_covariance
    end

    @static if VERSION < v"1.1"
        eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))
    end

    include("gels.jl")
    include("kf.jl")
    include("srkf.jl")
    include("sigmapoints.jl")
    include("augmented_sigmapoints.jl")
    include("ukf.jl")
    include("srukf.jl")
    include("aukf.jl")
    include("sraukf.jl")
    include("consider.jl")
    include("tests.jl")

end
