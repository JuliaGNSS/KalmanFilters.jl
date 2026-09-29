# The square root filters only need the upper triangular factor R of a QR
# decomposition. `geqrf` computes exactly that, whereas `gels` additionally applies
# Qᴴ to a right-hand side and solves a triangular system, which roughly doubles the
# cost. These wrappers take a preallocated workspace, so that the in-place updates
# don't allocate (`LAPACK.geqrf!` allocates its workspace on every call).
for (geqrf, elty) in (
    (:dgeqrf_, :Float64),
    (:sgeqrf_, :Float32),
    (:zgeqrf_, :ComplexF64),
    (:cgeqrf_, :ComplexF32),
)
    @eval begin
        #      SUBROUTINE DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, LDA, LWORK, M, N
        function mygeqrf!(
            res::AbstractMatrix{$elty},
            A::AbstractMatrix{$elty},
            tau::Vector{$elty},
            work::Vector{$elty},
        )
            Base.require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            k = min(m, n)
            if length(tau) < k
                throw(DimensionMismatch("tau has length $(length(tau)), but needs $k"))
            end
            info = Ref{BlasInt}()
            ccall(
                (@blasfunc($geqrf), liblapack),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                ),
                m,
                n,
                A,
                max(1, stride(A, 2)),
                tau,
                work,
                BlasInt(length(work)),
                info,
            )
            LAPACK.chklapackerror(info[])
            res .= @view(A[1:k, 1:k])
            triu!(res)
        end

        function calc_geqrf_working_size(A::AbstractMatrix{$elty})
            Base.require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            info = Ref{BlasInt}()
            tau = Vector{$elty}(undef, max(1, min(m, n)))
            work = Vector{$elty}(undef, 1)
            ccall(
                (@blasfunc($geqrf), liblapack),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                ),
                m,
                n,
                A,
                max(1, stride(A, 2)),
                tau,
                work,
                BlasInt(-1),
                info,
            )
            LAPACK.chklapackerror(info[])
            max(1, BlasInt(real(work[1])))
        end
    end
end

"""
    mygeqrf!(res, A, tau, work) -> res

Computes the QR decomposition of `A` with LAPACK's `geqrf` and writes the upper
triangular factor `R` into `res`. `A` is overwritten, `tau` must have at least
`min(size(A)...)` elements and `work` is the workspace, whose optimal length is given
by [`calc_geqrf_working_size`](@ref).
"""
mygeqrf!(res::AbstractMatrix, A::AbstractMatrix, tau::Vector, work::Vector)

"""
    calc_geqrf_working_size(A) -> working_size

Calculates the optimal workspace length of [`mygeqrf!`](@ref) for matrices of the size
and element type of `A`.
"""
calc_geqrf_working_size(A::AbstractMatrix)

# LAPACK's `geqrf` factorizes matrices with fewer than 128 columns unblocked, one
# Householder reflector at a time with `gemv`/`ger` calls. For more than a few dozen
# columns this is slow, and multithreaded OpenBLAS makes it slower still (2.5 to 5 times
# for 50 to 100 columns). `geqrt` and `tpqrt` instead use small blocks of
# `QR_BLOCK_SIZE` columns that are applied with matrix-matrix products, and `tpqrt` also
# skips the zeros of an upper triangular block. Both need `T` and `work` of length
# `qr_block_size(n) * n` for a matrix with `n` columns.
const QR_BLOCK_SIZE = 8

qr_block_size(num_cols) = max(1, min(QR_BLOCK_SIZE, num_cols))

for (geqrt, tpqrt, elty) in (
    (:dgeqrt_, :dtpqrt_, :Float64),
    (:sgeqrt_, :stpqrt_, :Float32),
    (:zgeqrt_, :ztpqrt_, :ComplexF64),
    (:cgeqrt_, :ctpqrt_, :ComplexF32),
)
    @eval begin
        #      SUBROUTINE DGEQRT( M, N, NB, A, LDA, T, LDT, WORK, INFO )
        function mygeqrt!(
            res::AbstractMatrix{$elty},
            A::AbstractMatrix{$elty},
            T::Vector{$elty},
            work::Vector{$elty},
        )
            Base.require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            k = min(m, n)
            nb = qr_block_size(k)
            check_qr_block_workspace(T, work, nb, n)
            info = Ref{BlasInt}()
            ccall(
                (@blasfunc($geqrt), liblapack),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{BlasInt},
                ),
                m,
                n,
                nb,
                A,
                max(1, stride(A, 2)),
                T,
                nb,
                work,
                info,
            )
            LAPACK.chklapackerror(info[])
            res .= @view(A[1:k, 1:k])
            triu!(res)
        end

        #      SUBROUTINE DTPQRT( M, N, L, NB, A, LDA, B, LDB, T, LDT, WORK, INFO )
        function mytpqrt!(
            res::AbstractMatrix{$elty},
            A::AbstractMatrix{$elty},
            T::Vector{$elty},
            work::Vector{$elty},
        )
            Base.require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            m >= n ||
                throw(DimensionMismatch("A must have at least as many rows as columns"))
            num_dense_rows = m - n
            nb = qr_block_size(n)
            check_qr_block_workspace(T, work, nb, n)
            info = Ref{BlasInt}()
            # LAPACK's A is the triangular block at the bottom and its B is the dense
            # block above; both are addressed in place with the leading dimension of `A`.
            ccall(
                (@blasfunc($tpqrt), liblapack),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{BlasInt},
                ),
                num_dense_rows,
                n,
                0,
                nb,
                @view(A[(num_dense_rows+1):m, :]),
                max(1, stride(A, 2)),
                A,
                max(1, stride(A, 2)),
                T,
                nb,
                work,
                info,
            )
            LAPACK.chklapackerror(info[])
            res .= @view(A[(num_dense_rows+1):m, :])
            triu!(res)
        end
    end
end

function check_qr_block_workspace(T, work, nb, n)
    if length(T) < nb * n
        throw(DimensionMismatch("T has length $(length(T)), but needs $(nb * n)"))
    end
    if length(work) < nb * n
        throw(DimensionMismatch("work has length $(length(work)), but needs $(nb * n)"))
    end
end

"""
    mygeqrt!(res, A, T, work) -> res

Computes the QR decomposition of `A` with LAPACK's blocked `geqrt` and writes the upper
triangular factor `R` into `res`. `A` is overwritten. `T` and `work` must have at least
`qr_block_size(size(A, 2)) * size(A, 2)` elements.
"""
mygeqrt!(res::AbstractMatrix, A::AbstractMatrix, T::Vector, work::Vector)

"""
    mytpqrt!(res, A, T, work) -> res

Computes the upper triangular factor `R` of the QR decomposition of `A`, whose last
`size(A, 2)` rows must form an upper triangular matrix, and writes it into `res`.
LAPACK's `tpqrt` exploits the zeros below the diagonal of that block. `A` is
overwritten. `T` and `work` must have at least `qr_block_size(size(A, 2)) * size(A, 2)`
elements.
"""
mytpqrt!(res::AbstractMatrix, A::AbstractMatrix, T::Vector, work::Vector)

"""
    householder_upper_triangular!(res, A, num_dense_rows) -> res

Computes the upper triangular factor `R` of the QR decomposition of `A` with Householder
reflections and writes it into `res`. Below its first `num_dense_rows` rows, `A` may
have an upper triangular block, whose zeros are skipped: column `k` is only nonzero up
to row `num_dense_rows + k`, and the reflections keep it that way. Pass
`num_dense_rows = size(A, 1)` for a dense `A`. `A` is overwritten.

For small matrices this is several times faster than LAPACK, whose routines apply every
reflection with separate BLAS calls.
"""
function householder_upper_triangular!(
    res::AbstractMatrix{T},
    A::AbstractMatrix{T},
    num_dense_rows::Integer,
) where {T<:Union{Real,Complex}}
    Base.require_one_based_indexing(A)
    m, n = size(A)
    m >= n || throw(DimensionMismatch("A must have at least as many rows as columns"))
    @inbounds for k = 1:n
        last_row = min(m, num_dense_rows + k)
        α = A[k, k]
        σ = zero(real(T))
        @simd for i = (k+1):last_row
            σ += abs2(A[i, k])
        end
        norm2 = abs2(α) + σ
        if floatmin(norm2) < norm2 < floatmax(norm2)
            x_norm = sqrt(norm2)
            is_reduced = iszero(σ)
        else
            # The squares under- or overflowed, compute the norm with scaling instead
            x = view(A, k:last_row, k)
            x_norm = norm(x)
            is_reduced = all(iszero, view(x, 2:length(x)))
        end
        # Column k is already reduced, the reflection is the identity
        is_reduced && isreal(α) && continue
        # Like LAPACK's `larfg`: `H' * x = β * e₁` with `H = I - τ * v * v'` and `v[1] = 1`
        β = -copysign(x_norm, real(α))
        τ = (β - α) / β
        scale = inv(α - β)
        @simd for i = (k+1):last_row
            A[i, k] *= scale
        end
        A[k, k] = β
        τ_conj = conj(τ)
        for j = (k+1):n
            s = A[k, j]
            @simd for i = (k+1):last_row
                s += conj(A[i, k]) * A[i, j]
            end
            s *= τ_conj
            A[k, j] -= s
            @simd for i = (k+1):last_row
                A[i, j] -= s * A[i, k]
            end
        end
    end
    res .= @view(A[1:n, 1:n])
    triu!(res)
end

# Below this number of elements, `householder_upper_triangular!` is faster than the
# blocked LAPACK routines (and also than `geqrf`). Complex arithmetic makes each element
# about four times as expensive, which shifts the break-even point accordingly.
use_native_qr(A::AbstractMatrix{<:Real}) = length(A) < 8192
use_native_qr(A::AbstractMatrix{<:Complex}) = length(A) < 2048

"""
    calc_upper_triangular_of_stacked_qr_inplace!(res, A, T, work) -> res

Writes the upper triangular factor `R` of the QR decomposition of `A` into `res`, where
the last `size(A, 2)` rows of `A` form an upper triangular matrix, e.g. the Cholesky
factor of a noise covariance below the weighted sigma points. Both
[`householder_upper_triangular!`](@ref), used for small matrices, and `tpqrt`, used for
larger ones, skip the zeros of the triangular block. `A` is overwritten. `T` and `work`
must have at least [`calc_qr_workspace_length`](@ref) elements.
"""
function calc_upper_triangular_of_stacked_qr_inplace!(res, A, T, work)
    if use_native_qr(A)
        householder_upper_triangular!(res, A, size(A, 1) - size(A, 2))
    else
        mytpqrt!(res, A, T, work)
    end
end

"""
    calc_upper_triangular_of_dense_qr_inplace!(res, A, T, work) -> res

Like [`calc_upper_triangular_of_stacked_qr_inplace!`](@ref), but for a dense `A`: large
matrices are factorized with `geqrt`.
"""
function calc_upper_triangular_of_dense_qr_inplace!(res, A, T, work)
    if use_native_qr(A)
        householder_upper_triangular!(res, A, size(A, 1))
    else
        mygeqrt!(res, A, T, work)
    end
end

"""
    calc_qr_workspace_length(A) -> length

Length of both the `T` and the `work` vector that
[`calc_upper_triangular_of_stacked_qr_inplace!`](@ref) and
[`calc_upper_triangular_of_dense_qr_inplace!`](@ref) need for matrices of the size of `A`.
"""
calc_qr_workspace_length(A::AbstractMatrix) = qr_block_size(size(A, 2)) * size(A, 2)

"""
    calc_upper_triangular_of_qr!(A, qr_inplace!) -> R

Allocating variant of the in-place QR decompositions `qr_inplace!`, e.g.
[`calc_upper_triangular_of_stacked_qr_inplace!`](@ref). Matrices whose element type
LAPACK doesn't support fall back to `calc_upper_triangular_of_qr!(A)`.
"""
function calc_upper_triangular_of_qr!(
    A::StridedMatrix{<:BlasFloat},
    qr_inplace!::F,
) where {F}
    n = size(A, 2)
    qr_inplace!(
        similar(A, n, n),
        A,
        similar(A, calc_qr_workspace_length(A)),
        similar(A, calc_qr_workspace_length(A)),
    )
end

calc_upper_triangular_of_qr!(A, qr_inplace!) = calc_upper_triangular_of_qr!(A)
