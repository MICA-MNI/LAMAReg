"""Regression tests for apply-warpfield interpolation handling."""

import sys
from unittest.mock import Mock

import lamareg.cli as cli
import lamareg.scripts.lamar as lamar_module


def test_cli_forwards_apply_warpfield_interpolation(monkeypatch):
    """The top-level command must not silently discard --interpolation."""
    mocked_lamareg = Mock()
    monkeypatch.setattr(cli, "lamareg", mocked_lamareg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lamareg",
            "apply-warpfield",
            "--moving",
            "moving.nii.gz",
            "--fixed",
            "fixed.nii.gz",
            "--output",
            "output.nii.gz",
            "--affine",
            "affine.mat",
            "--warpfield",
            "warp.nii.gz",
            "--interpolation",
            "nearestNeighbor",
        ],
    )

    cli.main()

    assert mocked_lamareg.call_args.kwargs["interpolation"] == "nearestNeighbor"


def test_apply_warpfield_forwards_interpolation_to_ants_wrapper(tmp_path, monkeypatch):
    """The workflow must pass the requested interpolator to apply_warp."""
    moving = tmp_path / "moving.nii.gz"
    fixed = tmp_path / "fixed.nii.gz"
    affine = tmp_path / "affine.mat"
    warp = tmp_path / "warp.nii.gz"
    output = tmp_path / "output.nii.gz"
    for path in (moving, fixed, affine, warp):
        path.touch()

    mocked_apply_warp = Mock()
    monkeypatch.setattr(lamar_module.ants, "image_read", lambda path: path)
    monkeypatch.setattr(lamar_module, "apply_warp", mocked_apply_warp)

    lamar_module.lamareg(
        input_image=str(moving),
        reference_image=str(fixed),
        output_image=str(output),
        apply_warpfield=True,
        affine_file=str(affine),
        warp_file=str(warp),
        interpolation="nearestNeighbor",
    )

    assert mocked_apply_warp.call_args.kwargs["interpolation"] == "nearestNeighbor"
