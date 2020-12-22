using Documenter
using DiscreteElasticRods

pages = [
    "Home" => "index.md",
    "Kinematics" => "kinematics.md",
    "Problem Specification" => "specification.md"
]

makedocs(
    sitename = "DiscreteElasticRods",
    authors = "James Gabbard",
    format = Documenter.HTML(),
    modules = [DiscreteElasticRods],
    pages = pages
)

deploydocs(
    deploy_config = Documenter.GitHubActions(),
    repo   = "github.com/jamesgabbard/DiscreteElasticRods.jl.git"
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
