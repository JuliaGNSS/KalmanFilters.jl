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
