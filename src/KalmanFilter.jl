module KalmanFilter

    export Weights, calc_sigma_points, kf_time_update, augment, ukf_measurement_update, ukf_time_update

    include("kf.jl")
    include("ukf.jl")

    function augment(x, P, R)
        x_a = [x; zeros(size(R, 1))]
        P_a = [P                            zeros(size(P,1),size(R,2));
               zeros(size(R,1),size(P,2))   R ]
        x_a, P_a
    end

    function augment(x, P, Q, R)
        augment(augment(x, P, Q)..., R)
    end

end
