"""The wristband (``clvhd.wristband``): connect, address and configure, against the simulated
Teensy."""

import pytest

import clvhd


class TestConnect:
    def test_connects_while_the_controller_boots(self, make_sim):
        # right after a power-up the Teensy answers only after its 1 s setup()
        s = make_sim(boot_s=1)
        with clvhd.Wristband(s.port, expected_modules=13) as wb:
            assert wb.firmware == (4, 1) and wb.ctrl.mirror()

    def test_connects_and_configures(self, sim):
        with clvhd.Wristband(sim.port, expected_modules=13) as wb:
            assert wb.nb_modules == 13
            assert wb.ctrl.mirror()
            cfg = wb.configure()
            assert abs(cfg.odr() - 2083.33) < 0.01
            assert wb.read_errors() == [""] * 13
            wb.configure(lead_off=True)  # LOD registers read back
            assert wb.set_spi_clock(8_000_000) == 8_000_000

    def test_wrong_module_count_raises(self, sim):
        with pytest.raises(clvhd.ControllerError, match="expected 11"):
            clvhd.Wristband(sim.port, expected_modules=11)
