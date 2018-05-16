function _time_update(𝐱, 𝐏, iweights, 𝐓::Matrix, 𝐐)
    𝐱_next = 𝐓 * 𝐱
    𝐏_next = 𝐓 * 𝐏 * 𝐓 + 𝐐
    𝐱_next, 𝐏_next
end