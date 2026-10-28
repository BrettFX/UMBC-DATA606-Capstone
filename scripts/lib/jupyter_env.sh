# Jupyter kernel registration for setup.sh (sourced, not run directly).
#
# Re-running setup.sh on an existing environment can leave pyzmq/ipykernel installed twice, once by pip (an earlier setup)
# and once by conda (`conda env update` adds its own copy but never removes pip's files). The mix imports fine until
# `import zmq` fails with "module 'zmq.backend.cython' has no attribute 'PYZMQ_DRAFT_API'". Registering the kernel is then
# impossible, but it is optional for the app, so these helpers repair what they can and otherwise warn instead of aborting.

jupyter_stack_ok() {  # $1 = conda env name
    conda run -n "$1" python -c "import zmq, ipykernel, jupyter_client" >/dev/null 2>&1
}

repair_jupyter_stack() {  # $1 = conda env name
    echo "The Jupyter packages in '$1' are broken (pyzmq/ipykernel installed by both pip and conda). Repairing..."
    # Remove pip's copy first so conda's reinstall does not land on top of stale files.
    conda run -n "$1" python -m pip uninstall -y pyzmq ipykernel >/dev/null 2>&1 || true
    conda install -y -q -n "$1" -c conda-forge --force-reinstall pyzmq ipykernel
}

# Register env $1 as a Jupyter kernel named $1 (re-registering is harmless). Never fails the caller: returns 1 after a warning.
register_jupyter_kernel() {
    local env="$1"
    if ! jupyter_stack_ok "$env"; then
        repair_jupyter_stack "$env" || true
    fi
    if jupyter_stack_ok "$env" && conda run -n "$env" python -m ipykernel install --user --name "$env" --display-name "$env"; then
        return 0
    fi
    echo "WARNING: could not register '$env' as a Jupyter kernel. The app and scripts are unaffected; only notebooks need it." >&2
    echo "         To fix it by hand:  conda activate $env && pip uninstall -y pyzmq ipykernel && conda install -c conda-forge --force-reinstall pyzmq ipykernel" >&2
    echo "         then:               python -m ipykernel install --user --name $env --display-name $env" >&2
    return 1
}
