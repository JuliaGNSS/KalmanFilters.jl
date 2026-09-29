function SRAUKFTUIntermediate(::Type{T}, num_x::Number) where {T}
    xi_temp = Vector{T}(undef, num_x)
    qr_A = Matrix{T}(undef, 4 * num_x, num_x)
    qr_tau = zeros(T, calc_qr_workspace_length(qr_A))
    qr_space_length = calc_qr_workspace_length(qr_A)
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
        qr_tau,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
    )
end

SRAUKFTUIntermediate(num_x::Number) = SRAUKFTUIntermediate(Float64, num_x)

function SRAUKFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T}
    qr_A = Matrix{T}(undef, 2 * num_x + 2 * num_y, num_y)
    qr_tau = zeros(T, calc_qr_workspace_length(qr_A))
    qr_space_length = calc_qr_workspace_length(qr_A)
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
        qr_tau,
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
    f!::F,
    Q::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
) where {F}
    time_update!(tu, x, Augmented(P, Q), f!, Q; weight_params = weight_params)
end

function measurement_update!(
    mu::SRUKFMUIntermediate,
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    y,
    h!::F,
    R::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
) where {F}
    measurement_update!(mu, x, Augmented(P, R), y, h!, R; weight_params = weight_params)
end
