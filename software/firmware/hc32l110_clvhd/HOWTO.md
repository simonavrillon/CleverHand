# HOWTO

Building and flashing over SWD with pyOCD (DAPLink) are in [docs/flashing.md](../../../docs/flashing.md), part 1. This file covers J-Link and debugging.

## Add HC32L110 device support to J-Link

J-Link doesn't support the HC32L110 out of the box:

1. Create a folder `HDSC` in the J-Link `Devices` folder (`/opt/SEGGER/JLink/Devices` on Linux, `/Applications/SEGGER/JLink/Devices` on macOS), and copy `Misc/Flash_Algorithms/HC32L110B4_C4.FLM` and `HC32L110B6_C6.FLM` into it.
2. In `JLinkDevices.xml`, in the same J-Link folder, add these lines before `</DataBase>` (`Misc/JLinkDevicesAddon.xml` has them, along with other HDSC parts):

   ```xml
   <Device>
     <ChipInfo Vendor="HDSC" Name="HC32L110x4" WorkRAMAddr="0x20000000" WorkRAMSize="0x800" Core="JLINK_CORE_CORTEX_M0"/>
     <FlashBankInfo Name="Flash_16K" BaseAddr="0x0" MaxSize="0x4000" Loader="Devices/HDSC/HC32L110B4_C4.FLM" LoaderType="FLASH_ALGO_TYPE_OPEN" AlwaysPresent="1"/>
   </Device>
   <Device>
     <ChipInfo Vendor="HDSC" Name="HC32L110x6" WorkRAMAddr="0x20000000" WorkRAMSize="0x1000" Core="JLINK_CORE_CORTEX_M0"/>
     <FlashBankInfo Name="Flash_32K" BaseAddr="0x0" MaxSize="0x8000" Loader="Devices/HDSC/HC32L110B6_C6.FLM" LoaderType="FLASH_ALGO_TYPE_OPEN" AlwaysPresent="1"/>
   </Device>
   ```

3. Flash with `make flash FLASH_PROGRM=jlink`. If `JLinkExe` isn't at `/opt/SEGGER/JLink/JLinkExe`, add `JLINKEXE=JLinkExe` (on the `PATH`) or its full path.

## Debug in VS Code

The build already has debug information and no optimisation (`DEBUG_FLAGS` and `OPT` in `rules.mk`). Install the Cortex-Debug extension, then add `.vscode/launch.json` (shown for a J-Link; for a DAPLink, use `"servertype": "pyocd"`, `"targetId": "hc32l110c6ua"` and `"serverArgs": ["--config", "${workspaceFolder}/Misc/pyocd.yaml"]` so pyOCD loads the HC32L110 pack):

```json
{
    "configurations": [
        {
            "name": "Cortex Debug",
            "cwd": "${workspaceFolder}",
            "executable": "${workspaceFolder}/Build/app.elf",
            "request": "launch",
            "type": "cortex-debug",
            "runToEntryPoint": "main",
            "servertype": "jlink",
            "device": "HC32L110x6",
            "interface": "swd",
            "svdFile": "${workspaceFolder}/Misc/HC32L110.svd"
        }
    ]
}
```

If VS Code doesn't find the tools, set `cortex-debug.gdbPath` (the toolchain's `arm-none-eabi-gdb`) and `cortex-debug.JLinkGDBServerPath` in `.vscode/settings.json`.

## About the HC32L110 driver

Part of the driver in `Libraries/HC32L110_Driver` has been heavily modified from HDSC's original (HC32L110_DDL_Rev1.1.4): the interrupt callbacks are wired to the vector table directly, and some peripheral functions are macros for speed.

## References

- Template: [IOsetting/hc32l110-template](https://github.com/IOsetting/hc32l110-template)
- HDSC product page: [hdsc.com.cn](https://www.hdsc.com.cn/Category82)
- Jeroen Domburg's HC32L110 SDK for GCC/GDB/OpenOCD: [Spritetm/hc32l110-gcc-sdk](https://github.com/Spritetm/hc32l110-gcc-sdk)
