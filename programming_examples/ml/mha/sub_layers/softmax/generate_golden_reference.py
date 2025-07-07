import torch

def main():
    N = 262144 
    TILE_SIZE = 1024
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A = (torch.rand((N // TILE_SIZE, TILE_SIZE), dtype=torch.float32) * 12 - 4).to(dtype)

    B = torch.empty_like(A)
    B = torch.nn.functional.softmax(A.to(torch.bfloat16), dim=-1).to(dtype)

    # Convert to float32 and to Python lists for dumping them into headers
    A_list = A.to(torch.float32).flatten().tolist()
    B_list = B.to(torch.float32).flatten().tolist()

    # Export to header
    with open("golden_reference.h", "w") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n")
        f.write("constexpr int N = {};\n".format(N))
        f.write("constexpr int TILE_SIZE = {};\n".format(TILE_SIZE))
        f.write("constexpr std::array<float, N> golden_input = {")
        f.write(",".join(map(str, A_list)))
        f.write("};\n")
        f.write("constexpr std::array<float, N> golden_output = {")
        f.write(",".join(map(str, B_list)))
        f.write("};\n")

if __name__ == "__main__":
    main()