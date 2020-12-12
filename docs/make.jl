using Documenter
using DiscreteElasticRods

makedocs(
    sitename = "DiscreteElasticRods",
    format = Documenter.HTML(),
    modules = [DiscreteElasticRods]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
