# COM_MOD firmware (HC32L110)

Firmware of the COM_MOD, the HC32L110C6UA on each node of the CleverHand wristband. It:

- takes the node's address at power-up, from its position in the chain, and drives the 5 address lines of the node's 74HC688, which selects the ADS1293 when the Teensy puts that address on the bus. See [initialisation/README.md](initialisation/README.md);
- then runs an I2C slave at that address and drives the node's two WS2812 LEDs. Each I2C write is 4 bytes: `led` (1 or 2), `r`, `g`, `b`.

The EMG data doesn't go through the COM_MOD: the Teensy reads the ADS1293 directly over SPI.

## Build and flash

`make` builds `Build/app.hex`, and `make flash` loads it over SWD with pyOCD (or a J-Link with `FLASH_PROGRM=jlink`). The wiring, the tools and the troubleshooting are in [docs/flashing.md](../../../docs/flashing.md), part 1. [HOWTO.md](HOWTO.md) covers J-Link device support and debugging in VS Code.

## Files

| Path | Content |
|---|---|
| `User/main.c` | the COM_MOD firmware |
| `initialisation/` | addressing protocol, with diagrams and scope captures |
| `Libraries/HC32L110_Driver/` | HDSC peripheral drivers (modified, see HOWTO.md) |
| `Libraries/CMSIS/` | Cortex-M0+ core headers, HC32L110 registers, startup code |
| `Libraries/LDScripts/` | linker scripts for the 16 KB (`x4`) and 32 KB (`x6`) parts |
| `Misc/` | pyOCD pack and config, J-Link flash algorithms and script, SVD file |
| `Makefile`, `rules.mk` | build and flash |

Built on the [IOsetting hc32l110-template](https://github.com/IOsetting/hc32l110-template) (Apache 2.0, see `LICENSE`).
