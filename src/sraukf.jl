function SRAUKFTUIntermediate(T::Type, num_x::Number)
    xi_temp = Vector{T}(undef, num_x)
    qr_zeros = zeros(T, 4 * num_x)
    qr_A = Matrix{T}(undef, 4 * num_x, num_x)
    qr_space_length = calc_gels_working_size(qr_A, qr_zeros)
    SRUKFTUIntermediate(
        Augmented(Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x)),
        Augmented(xi_temp, xi_temp),
        xi_temp,
        TransformedSigmaPoints(
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_x, 4 * num_x),
            MeanSetWeightingParameters(0.0),
        ), # Weighting parameters will be reset
        TransformedSigmaPoints(
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_x, 4 * num_x),
            MeanSetWeightingParameters(0.0),
        ),
        qr_zeros,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
    )
end

SRAUKFTUIntermediate(num_x::Number) = SRAUKFTUIntermediate(Float64, num_x)

function SRAUKFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    qr_zeros = zeros(T, 2 * num_x + 2 * num_y)
    qr_A = Matrix{T}(undef, 2 * num_x + 2 * num_y, num_y)
    qr_space_length = calc_gels_working_size(qr_A, qr_zeros)
    SRUKFMUIntermediate(
        Augmented(Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_y, num_y)),
        Augmented(Vector{T}(undef, num_x), Vector{T}(undef, num_y)),
        Vector{T}(undef, num_y),
        Vector{T}(undef, num_y),
        TransformedSigmaPoints(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, 2 * num_x + 2 * num_y),
            MeanSetWeightingParameters(0.0),
        ), # Weighting parameters will be reset
        TransformedSigmaPoints(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, 2 * num_x + 2 * num_y),
            MeanSetWeightingParameters(0.0),
        ),
        Vector{T}(undef, num_y),
        qr_zeros,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
        Vector{T}(undef, num_x),
    )
end

SRAUKFMUIntermediate(num_x::Number, num_y::Number) =
    SRAUKFMUIntermediate(Float64, num_x, num_y)

function time_update!(
    tu::SRUKFTUIntermediate,
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    f!,
    Q::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
)
    time_update!(tu, x, Augmented(P, Q), f!, Q; weight_params = weight_params)
end

function measurement_update!(
    mu::SRUKFMUIntermediate,
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    y,
    h!,
    R::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
)
    measurement_update!(mu, x, Augmented(P, R), y, h!, R; weight_params = weight_params)
end
