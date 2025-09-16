using Documenter
using QuantumReservoirComputing

makedocs(;
    sitename="QuantumReservoirComputing",
    modules=[QuantumReservoirComputing],
    remotes=nothing,
    pages=["Home" => "index.md", "TestDocs" => "docstrings.md"],
)
