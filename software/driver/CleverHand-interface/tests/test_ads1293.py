"""The ADS1293 configuration (``clvhd.ads1293``)."""

import pytest

import clvhd
from clvhd import ads1293


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        clvhd.Config(R2=7).validate()
    with pytest.raises(ValueError):
        clvhd.Config(enable=[False, False, False]).validate()


def test_config_rejects_a_missing_rld_module():
    # checked before anything is sent to the controller
    with pytest.raises(ValueError, match="rld_module"):
        ads1293.configure(None, 13, clvhd.Config(rld_module=13))
    with pytest.raises(ValueError, match="rld_input"):
        clvhd.Config(rld_input=7).validate()


def test_lead_off_registers_and_flags():
    regs = dict(ads1293.register_values(clvhd.Config(lead_off=True)))
    assert (regs[ads1293.LOD_CN], regs[ads1293.LOD_EN], regs[ads1293.LOD_CURRENT]) == (0x00, 0x1F, 1)
    assert dict(ads1293.register_values(clvhd.Config()))[ads1293.LOD_CN] == 0x08  # shut down
    assert ads1293.describe_errors([0x10, 0x08, 0, 0, 0, 0, 0]) == "lead off, IN5 lead off"
