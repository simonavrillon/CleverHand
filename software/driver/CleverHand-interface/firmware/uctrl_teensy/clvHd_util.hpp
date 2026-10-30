#include <SPI.h>
#include <Wire.h>

#define READ 0b10000000
#define REG_MASK 0b01111111

// Address 16 (bit 4 set, bits 0-3 clear) is never used by a module: the bus
// idles there, and bit 4 is raised while bits 0-3 change so no other module
// is selected by a transient address. This limits the bus to 15 modules,
// which the 1-byte reply length already imposes.
#define CLVHD_IDLE_ADDR 0x10
#define CLVHD_MAX_MODULES 15

#define CLVHD_INIT_OK 0
#define CLVHD_INIT_STUCK 1     // a module never took an address (broken IO1 chain)
#define CLVHD_INIT_TOO_MANY 2  // more modules than CLVHD_MAX_MODULES

//class used to unify the different functions used to interact with the EMG modules
class ClvHd
{
    public:
    // Initialise the SPI bus and the selction pins
    ClvHd() {};
    ~ClvHd() {};

    void
    begin()
    {
        SPI.begin();
        for(int i = 0; i < 5; i++) pinMode(m_addPins[i], OUTPUT);
        deselect();
        pinMode(m_availPin, INPUT_PULLUP);
        pinMode(m_clkPin, OUTPUT); //CLK
        pinMode(m_outPin, OUTPUT); //pull LOW to stop counting
        digitalWrite(m_outPin, LOW);
        digitalWrite(m_clkPin, LOW);

        this->m_nbModule = 0;
    };

    /**
     * @brief Set the SPI clock used for every module access.
     *
     * @return uint32_t the clock that was set
     */
    uint32_t
    setSpiClock(uint32_t hz)
    {
        if(hz < 100000)
            hz = 100000;
        if(hz > 20000000) // ADS1293 maximum
            hz = 20000000;
        m_spiSettings = SPISettings(hz, MSBFIRST, SPI_MODE0);
        return hz;
    }

    /**
     * @brief Initialise the modules and return the number of module found.
     *
     * @param status set to CLVHD_INIT_OK, CLVHD_INIT_STUCK or CLVHD_INIT_TOO_MANY
     * @return uint8_t number of module found
     */
    uint8_t
    initModules(uint8_t *status = nullptr)
    {
        if(initialized)
        {
            if(status != nullptr)
                *status = m_initStatus;
            return this->m_nbModule;
        }

        int width = 1;
        int pulses = 0;
        //send pulses as long as the m_availPin is pulled low by an unaddressed module
        while(digitalRead(m_availPin) == 0 && pulses < 32)
        {
            digitalWrite(m_clkPin, HIGH);
            delayMicroseconds(1);
            digitalWrite(m_clkPin, LOW);
            delayMicroseconds(1);
            pulses++;
            delay(width);
        }
        if(digitalRead(m_availPin) == 0)
        {
            // a module never saw its STOP_IN go low: how many modules got an
            // address before it is unknown, so report none
            m_initStatus = CLVHD_INIT_STUCK;
            this->m_nbModule = 0;
        }
        else if(pulses > CLVHD_MAX_MODULES)
        {
            m_initStatus = CLVHD_INIT_TOO_MANY;
            this->m_nbModule = CLVHD_MAX_MODULES;
        }
        else
        {
            m_initStatus = CLVHD_INIT_OK;
            this->m_nbModule = pulses;
        }
        // one more rising SCL edge, with every module addressed (m_availPin
        // high), tells the modules to start their I2C slave. Drive it and
        // leave SCL high (the I2C idle level): the modules can miss the edge
        // Wire.begin() makes when it takes the pin over.
        digitalWrite(m_clkPin, HIGH);
        delay(width);
        Wire.begin();
        initialized = true;
        if(status != nullptr)
            *status = m_initStatus;
        return this->m_nbModule;
    }

    /**
     * @brief Select the module by activating the coreponding address pins.
     *
     * @param id Address of the module to select (1 to 15)
     */
    inline void
    selectBrd(uint8_t id)
    {
        digitalWriteFast(6, HIGH); // guard bit: no module matches while bits 0-3 change
        digitalWriteFast(2, id & 0x01);
        digitalWriteFast(3, id & 0x02);
        digitalWriteFast(4, id & 0x04);
        digitalWriteFast(5, id & 0x08);
        if(!(id & 0x10))
            digitalWriteFast(6, LOW);
        delayNanoseconds(100); // 74HC688 propagation at 3.3 V
    }

    inline void
    deselect()
    {
        delayNanoseconds(50); // CSB hold
        selectBrd(CLVHD_IDLE_ADDR);
    }

    /**
     * @brief Read n uint8_ts starting from the reg address of the module coreponding to id.
     *
     * @param n_cmd Number of uint8_ts to read from the module
     * @param cmd Array of uint8_ts to send to the module
     * @param val Array of uint8_ts to store the read values
     * @param n Number of uint8_ts to read from the module
     * @param id Address of the module to read from (1 to 15)
     */
    void
    readCmd(uint8_t n_cmd, uint8_t cmd[], uint8_t n, uint8_t val[], uint8_t id)
    {
        SPI.beginTransaction(m_spiSettings);
        selectBrd(id);
        for(unsigned i = 0; i < n_cmd; i++) SPI.transfer(cmd[i]);
        for(unsigned i = 0; i < n; i++) *(val + i) = SPI.transfer(0x00);
        deselect();
        SPI.endTransaction();
    }

    /**
     * @brief Write n uint8_t to the reg ad the module coreponding to id.
     *
     * @param n_cmd Number of uint8_ts to write to the module
     * @param cmd Array of uint8_ts to send to the module
     * @param n Number of uint8_ts to write to the module
     * @param val Array of uint8_ts to write to the module
     * @param id Address of the module to write to (1 to 15)
     */
    void
    writeCmd(uint8_t n_cmd, uint8_t cmd[], uint8_t n, uint8_t val[], uint8_t id)
    {
        SPI.beginTransaction(m_spiSettings);
        selectBrd(id);
        for(unsigned i = 0; i < n_cmd; i++) SPI.transfer(cmd[i]);
        for(unsigned i = 0; i < n; i++) SPI.transfer(val[i]);
        deselect();
        SPI.endTransaction();
    }

    uint8_t
    nbModules()
    {
        return m_nbModule;
    }

    private:
    bool initialized = false;
    uint8_t m_initStatus = CLVHD_INIT_OK;
    const int m_addPins[5] = {2, 3, 4, 5, 6};
    int m_nbModule = 0;
    int m_availPin = 18;
    int m_clkPin = 19;
    int m_outPin = 8;
    SPISettings m_spiSettings = SPISettings(4000000, MSBFIRST, SPI_MODE0);
};
