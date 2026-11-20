import math


def attention_tile_bytes(br, bc, d, bytes_per_elem):
    elements = br * d + 2 * bc * d + br * bc + br * d + 2 * br
    return elements * bytes_per_elem


def largest_block(sram_bytes, d, bytes_per_elem):
    bound = math.isqrt(sram_bytes // bytes_per_elem)
    if bound == 0:
        return 0

    b = 2 ** (bound.bit_length() - 1)
    while b > 0:
        if attention_tile_bytes(b, b, d, bytes_per_elem) <= sram_bytes:
            return b
        b //= 2

    return 0


def standard_attention_traffic(n, d, bytes_per_elem):
    elements = 5 * n * d + 4 * n * n
    return elements * bytes_per_elem


def tiled_attention_traffic(n, d, br, bytes_per_elem):
    query_blocks = (n + br - 1) // br
    query_and_output_elements = 2 * n * d
    key_and_value_elements = query_blocks * 2 * n * d
    return (query_and_output_elements + key_and_value_elements) * bytes_per_elem


def traffic_ratio(n, d, br, bytes_per_elem):
    standard_bytes = standard_attention_traffic(n, d, bytes_per_elem)
    tiled_bytes = tiled_attention_traffic(n, d, br, bytes_per_elem)
    return round(standard_bytes / tiled_bytes, 3)