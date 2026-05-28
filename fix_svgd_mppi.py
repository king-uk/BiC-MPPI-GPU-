def extract_func(text, func_name):
    start = text.find(func_name)
    if start == -1: return ""
    brace_idx = text.find('{', start)
    if brace_idx == -1: return ""
    
    count = 1
    idx = brace_idx + 1
    while count > 0 and idx < len(text):
        if text[idx] == '{': count += 1
        elif text[idx] == '}': count -= 1
        idx += 1
    return text[start:idx]

with open("mppi/cuda/bi_mppi_gpu.cu", "r") as f:
    text = f.read()

funcs = [
    "void BiMPPI_GPU::dbscan",
    "void BiMPPI_GPU::calculateU",
    "void BiMPPI_GPU::selectConnection",
    "void BiMPPI_GPU::concatenate",
    "void BiMPPI_GPU::partitioningControl",
    "void BiMPPI_GPU::solve",
    "void BiMPPI_GPU::move"
]

extracted = []
for fn in funcs:
    func_str = extract_func(text, fn)
    func_str = func_str.replace("BiMPPI_GPU::", "SVGDMPPI_GPU::")
    extracted.append(func_str)

with open("mppi/cuda/svgd_mppi_gpu.cu", "r") as f:
    svgd = f.read()

marker = "// Emptied unused functions"
if marker in svgd:
    svgd = svgd[:svgd.find(marker)]

svgd += marker + "\n" + "\n\n".join(extracted) + "\n"

with open("mppi/cuda/svgd_mppi_gpu.cu", "w") as f:
    f.write(svgd)

print("Properly replaced functions in svgd_mppi_gpu.cu")
