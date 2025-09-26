from setuptools import setup, find_packages

# install modules with 
#   `pip install .` 
# or for developers 
#   `pip install . [dev]`
setup(
    name='mcp-risk-server',
    version='1.0',
    description='MCP Risk Server package',
    author='Marie Hoffmann',
    author_email='aieoa-dev@proton.me',
    packages=find_packages(),
    install_requires=[
        'fastmcp',
        'git',
        'jinja2',
        'numpy',
        'openai',
        'pandas',
        'perplexityai',
        'pyaml',
        'regex',
        'requests',
        'pandas'
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "krippendorff"
        ]
    }
)
