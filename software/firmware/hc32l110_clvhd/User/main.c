#include "gpio.h"
#include "i2c.h"
#include "rgb.h"

// pin 34, 33, 32, 03, 15 are used for address output
// these pins can be found on the hc32l110 datasheet
// pin numbers 1-20, which shows their location on the
// controller are labelled on the diagram. The table
// underneath shows the pin name for each pin (e.g. pin
// number 2 is labelled as P01). In this case, pin number
// 18 (P34), 17 (P33), 16 (P32), 7 (P03), and 8 (P15) are
// used. The first digit in the pin name refers to the
// port whereas the second digit refers to the pin. Port
// is essentially a group of pins.

#define CLVHD_ADD0_PORT 3
#define CLVHD_ADD0_PIN 4
#define CLVHD_ADD1_PORT 3
#define CLVHD_ADD1_PIN 3
#define CLVHD_ADD2_PORT 3
#define CLVHD_ADD2_PIN 2
#define CLVHD_ADD3_PORT 0
#define CLVHD_ADD3_PIN 3
#define CLVHD_ADD4_PORT 1
#define CLVHD_ADD4_PIN 5

// I2C on pins SDA: 35, SCL: 36
#define CLVHD_SDA_PORT 3 // available pin
#define CLVHD_SDA_PIN 5
#define CLVHD_SCL_PORT 3 // CLK
#define CLVHD_SCL_PIN 6

#define CLVHD_STOP_IN_PORT 3
#define CLVHD_STOP_IN_PIN 1
#define CLVHD_STOP_OUT_PORT 0
#define CLVHD_STOP_OUT_PIN 2

#define HIGH 1
#define LOW 0

// Default LED brightness (0-255). Each WS2812 channel draws about 5 mA at 255,
// so full brightness on 12 nodes overloads the Teensy's 3.3 V supply.
#define LED_ON 16

volatile uint8_t r1 = 0, g1 = 0, b1 = 0;
volatile uint8_t r2 = 0, g2 = 0, b2 = 0;
uint8_t module_address = 0x01;

volatile uint8_t edge_seen = 0; // an addressing clock edge arrived since the last LED refresh

volatile uint8_t address_count = 1;
volatile uint8_t set = 0;
volatile uint8_t full_set = 0;

void Gpio_SetIO(uint8_t port, uint8_t pin, uint8_t value)
{
    if (value)
        GPIO_SetPinOutHigh(port, pin);
    else
        GPIO_SetPinOutLow(port, pin);
}

void set_address(uint8_t address)
{
    Gpio_SetIO(CLVHD_ADD0_PORT, CLVHD_ADD0_PIN, address & 0x01);        // set the first digit of the address (binary)
    Gpio_SetIO(CLVHD_ADD1_PORT, CLVHD_ADD1_PIN, (address >> 1) & 0x01); // set the second digit of the address (binary)
    Gpio_SetIO(CLVHD_ADD2_PORT, CLVHD_ADD2_PIN, (address >> 2) & 0x01); // set the third digit of the address (binary)
    Gpio_SetIO(CLVHD_ADD3_PORT, CLVHD_ADD3_PIN, (address >> 3) & 0x01); // set the fourth digit of the address (binary)
    Gpio_SetIO(CLVHD_ADD4_PORT, CLVHD_ADD4_PIN, (address >> 4) & 0x01); // set the fifth digit of the address (binary)
    module_address = address;
}

uint32_t i2cRxLen = 0;
uint8_t i2cRxData[4];

void handleI2C_slave()
{
    switch (I2C_GetState()) // get i2c state
    {
    case 0x60:        // ModuleAddress+W received, ACK returned
    case 0x70:        // GeneralCall+W received, ACK returned
        i2cRxLen = 0; // prepare for receiving data
        break;
    case 0x80: // data received after ModuleAddress+W, ACK returned
    case 0x90: // data received after GeneralCall+W, ACK returned
    {
        uint8_t byte = I2C_ReadByte(); // data received
        if (i2cRxLen < sizeof(i2cRxData))
            i2cRxData[i2cRxLen] = byte;
        i2cRxLen++; // only an exactly 4-byte message is applied below
        break;
    }
    case 0xA0: // STOP or repeated START condition received
        break;
    case 0xA8: // ModuleAddress+R received, ACK returned
    case 0xB8: // data transmitted, ACK received
    case 0xC0: // data transmitted, NACK received
    case 0xC8: // last data transmitted, ACK received
        I2C_WriteByte(0); // nothing to read (the Teensy only writes): answer 0 so a read cannot stall the bus
        break;
    case 0xF8: // no relevant state information available; SI = 0
        break;
    default:
        break;
    }

    if (i2cRxLen == 4)
    {
        switch (i2cRxData[0])
        {
        case 0x01:
            r1 = i2cRxData[1];
            g1 = i2cRxData[2];
            b1 = i2cRxData[3];
            break;
        case 0x02:
            r2 = i2cRxData[1];
            g2 = i2cRxData[2];
            b2 = i2cRxData[3];
            break;
        default:
            break;
        }
    }
    I2C_ClearIrq(); // clear interrupt flag
}

void init_i2c_slave(uint8_t address)
{
    // clock is on internal high speed 24MHz
    stc_i2c_config_t stcI2cCfg;
    Gpio_InitIOExt(CLVHD_SDA_PORT, CLVHD_SDA_PIN, GpioDirOut, FALSE, FALSE, TRUE, FALSE);
    Gpio_InitIOExt(CLVHD_SCL_PORT, CLVHD_SCL_PIN, GpioDirIn, FALSE, FALSE, TRUE, FALSE);
    Gpio_SetFunc_I2C_DAT_P35();
    Gpio_SetFunc_I2C_CLK_P36();

    CLK_EnablePeripheralClk(ClkPeripheralI2c);
    stcI2cCfg.enFunc = I2cBaud_En;
    stcI2cCfg.stcSlaveAddr.Addr = address;
    stcI2cCfg.stcSlaveAddr.Gc = 0; // no general call
    stcI2cCfg.u8Tm = 29;           // 100000Hz=(24000000/(8*(1+29)))
    stcI2cCfg.pfnI2cCb = handleI2C_slave;
    stcI2cCfg.bTouchNvic = 1; // set to TRUE if you want to use interrupts
    I2C_DeInit();
    I2C_Init(&stcI2cCfg);
    EnableNvic(I2C_IRQn, 3, TRUE);
    I2C_SetFunc(I2cAck_En);
    I2C_SetFunc(I2cMode_En);
}

void PORT3_IRQHandler(void)
{
    /** Clear interrupt flag before handling, otherwise it will be triggered twice */
    Gpio_ClearIrq(CLVHD_SCL_PORT, CLVHD_SCL_PIN);
    edge_seen = 1;
    if (set == 0)
    {
        if (GPIO_GetPinIn(CLVHD_STOP_IN_PORT, CLVHD_STOP_IN_PIN)) // previous module not addressed yet
        {
            r1 = LED_ON;
            g1 = LED_ON;
            b1 = 0;
            address_count++;
        }
        else
        {
            set_address(address_count);                                                      // set address
            set = 1;                                                                         // module initialised
            Gpio_InitIOExt(CLVHD_SDA_PORT, CLVHD_SDA_PIN, GpioDirIn, TRUE, FALSE, FALSE, 0); // set available pin to be INPUT
            // The next module handles this same clock edge in its own interrupt,
            // possibly late. Wait before lowering STOP_OUT so it still reads the
            // old (high) level on this edge and only takes an address on the next.
            delay100us(1);
            GPIO_SetPinOut(CLVHD_STOP_OUT_PORT, CLVHD_STOP_OUT_PIN, LOW); // set stop state for next module
            r2 = LED_ON;
            g2 = 0;
            b2 = LED_ON;
        }
    }
    else if (full_set == 0)
    {
        // check if all modules are initialised
        if (GPIO_GetPinIn(CLVHD_SDA_PORT, CLVHD_SDA_PIN))
        {
            full_set = 1;
            r2 = LED_ON;
            g2 = LED_ON;
            b2 = LED_ON;
            if (address_count == 1)
            {
                r1 = 0;
                g1 = LED_ON;
                b1 = 0;
            }
            // disable interrupt
            Gpio_DisableIrq(CLVHD_SCL_PORT, CLVHD_SCL_PIN, GpioIrqRising);
            // change usage of GPIO to I2C
            init_i2c_slave(module_address);
        }
    }
    else
    {
        // should not reach here
        r1 = LED_ON;
        g1 = LED_ON;
        b1 = 0;
        r2 = 0;
        g2 = LED_ON;
        b2 = 0;
    }
}

void init_process_initialisation()
{
    // stop state pins (input and output)
    // Pull-up: if the link from the previous module is broken, this module
    // never takes an address (the Teensy then reports a stuck chain) instead
    // of taking the same address as another module.
    Gpio_InitIOExt(CLVHD_STOP_IN_PORT, CLVHD_STOP_IN_PIN, GpioDirIn, TRUE, FALSE, FALSE, FALSE);    // Stop input
    Gpio_InitIOExt(CLVHD_STOP_OUT_PORT, CLVHD_STOP_OUT_PIN, GpioDirOut, TRUE, FALSE, FALSE, FALSE); // Stop output
    GPIO_SetPinOut(CLVHD_STOP_OUT_PORT, CLVHD_STOP_OUT_PIN, HIGH);                                  // set stop state

    // clk interrupt input
    Gpio_InitIOExt(CLVHD_SCL_PORT, CLVHD_SCL_PIN, GpioDirIn, FALSE, TRUE, FALSE, 0); // CLK interrupt pin
    Gpio_ClearIrq(CLVHD_SCL_PORT, CLVHD_SCL_PIN);
    Gpio_EnableIrq(CLVHD_SCL_PORT, CLVHD_SCL_PIN, GpioIrqRising);
    EnableNvic(PORT3_IRQn, DDL_IRQ_LEVEL_DEFAULT, TRUE);

    // available pin
    Gpio_InitIOExt(CLVHD_SDA_PORT, CLVHD_SDA_PIN, GpioDirOut, FALSE, FALSE, FALSE, FALSE);
    GPIO_SetPinOutLow(CLVHD_SDA_PORT, CLVHD_SDA_PIN);
}

void CLVHD_ConfigPin(void)
{
    // configure P01 to control the neopixel
    RGB_Init();
    RGB_Reset();
    r1 = LED_ON;
    g1 = 0;
    b1 = 0;
    r2 = 0;
    g2 = 0;
    b2 = 0;

    // set the address output pin as output
    Gpio_InitIOExt(CLVHD_ADD0_PORT, CLVHD_ADD0_PIN, GpioDirOut, FALSE, FALSE, FALSE, FALSE);
    Gpio_InitIOExt(CLVHD_ADD1_PORT, CLVHD_ADD1_PIN, GpioDirOut, FALSE, FALSE, FALSE, FALSE);
    Gpio_InitIOExt(CLVHD_ADD2_PORT, CLVHD_ADD2_PIN, GpioDirOut, FALSE, FALSE, FALSE, FALSE);
    Gpio_InitIOExt(CLVHD_ADD3_PORT, CLVHD_ADD3_PIN, GpioDirOut, FALSE, FALSE, FALSE, FALSE);
    Gpio_InitIOExt(CLVHD_ADD4_PORT, CLVHD_ADD4_PIN, GpioDirOut, FALSE, FALSE, FALSE, FALSE);

    init_process_initialisation();
}

int main(void)
{
    CLVHD_ConfigPin();

    while (1)
    {
        // Writing the LEDs disables interrupts for ~30 us, so skip it while
        // the addressing clock is running to keep the edge handling prompt.
        if (!full_set && edge_seen)
        {
            edge_seen = 0;
            delay1ms(5);
            continue;
        }
        RGB_SetColor(r1, g1, b1);
        RGB_SetColor(r2, g2, b2);
        delay1ms(10);
    }

    return 0;
}