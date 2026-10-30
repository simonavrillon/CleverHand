#define VERSION_MAJOR 4
#define VERSION_MINOR 1
#include "clvHd_util.hpp"

// ADS1293 registers used by the streaming loop
#define ADS_CONFIG_REG 0x00
#define ADS_DATA_STATUS_REG 0x30
#define ADS_DATA_CH1_ECG_REG 0x37
#define ADS_START_CONV 0x01
#define ADS_STANDBY 0x02

// Streaming frame: sync | type | module | counter(4) | t_us(4) | status | ecg(9) | crc8
#define FRAME_SYNC 0xA5
#define FRAME_DATA 'D'
#define FRAME_END 'E'
#define FRAME_SIZE 22
// time from the first sign of a new sample to its read: the channels of a
// module update up to ~90 us apart
#define STREAM_SETTLE_US 100

uint8_t recv_buff[64];
uint8_t send_buff[1024];
uint64_t *timestamp = (uint64_t *)send_buff;
uint8_t *size_buff = send_buff + 8;
uint8_t *vals_buff = send_buff + 9;
int i, n, n_cmd;
uint32_t mask_id = 0;

#define RECV_PAYLOAD_MAX (sizeof(recv_buff) - 7)
#define SEND_PAYLOAD_MAX (sizeof(send_buff) - 9)

ClvHd clvHd;

// streaming state
bool streaming = false;
uint32_t stream_mask = 0;
uint8_t stream_drdy = 0;
uint32_t stream_count[CLVHD_MAX_MODULES];
uint8_t stream_last[CLVHD_MAX_MODULES][9];   // ECG bytes of the last sample sent
bool stream_pending[CLVHD_MAX_MODULES];      // new sample seen, waiting to settle
uint32_t stream_seen_us[CLVHD_MAX_MODULES];  // when it was seen
uint8_t stream_status[CLVHD_MAX_MODULES];    // DATA_STATUS when it was seen

uint8_t
crc8(const uint8_t *data, int len)
{
    uint8_t crc = 0;
    for(int k = 0; k < len; k++)
    {
        crc ^= data[k];
        for(int b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
    return crc;
}

void
sendFrame(uint8_t type, uint8_t module, uint32_t counter, uint32_t t_us,
          uint8_t status, const uint8_t *ecg)
{
    uint8_t frame[FRAME_SIZE];
    frame[0] = FRAME_SYNC;
    frame[1] = type;
    frame[2] = module;
    memcpy(frame + 3, &counter, 4);
    memcpy(frame + 7, &t_us, 4);
    frame[11] = status;
    if(ecg != nullptr)
        memcpy(frame + 12, ecg, 9);
    else
        memset(frame + 12, 0, 9);
    frame[21] = crc8(frame + 1, FRAME_SIZE - 2);
    Serial.write(frame, FRAME_SIZE);
}

void
writeReg(uint8_t module, uint8_t reg, uint8_t value)
{
    uint8_t cmd = reg & REG_MASK;
    clvHd.writeCmd(1, &cmd, 1, &value, module + 1);
}

void
startStreaming()
{
    // the data registers still hold the last sample of the previous stream:
    // take it as the reference, so it is not sent as a new sample
    uint8_t data_cmd = ADS_DATA_CH1_ECG_REG | READ;
    for(i = 0; i < clvHd.nbModules(); i++)
    {
        stream_count[i] = 0;
        stream_pending[i] = false;
        if(stream_mask & ((uint32_t)1 << i))
            clvHd.readCmd(1, &data_cmd, 9, stream_last[i], i + 1);
    }
    // start the conversions back to back so the modules begin as close
    // together as the bus allows
    for(i = 0; i < clvHd.nbModules(); i++)
        if(stream_mask & ((uint32_t)1 << i))
            writeReg(i, ADS_CONFIG_REG, ADS_START_CONV);
    streaming = true;
}

void
stopStreaming()
{
    streaming = false;
    for(i = 0; i < clvHd.nbModules(); i++)
        if(stream_mask & ((uint32_t)1 << i))
            writeReg(i, ADS_CONFIG_REG, ADS_STANDBY);
    sendFrame(FRAME_END, 0, 0, micros(), 0, nullptr);
    Serial.send_now();
}

// 'i' payload: id | led | r | g | b, forwarded over I2C to the COM_MOD at id.
// At 100 kHz the transfer takes ~0.5 ms, so while streaming it can cost
// about one sample per module.
void
setLed()
{
    Serial.readBytes((char *)recv_buff + 1, 5);
    Wire.beginTransmission(recv_buff[1]);
    Wire.write(recv_buff + 2, 4);
    Wire.endTransmission();
}

// Poll every streamed module and forward each new ECG sample.
// DATA_STATUS alone misses samples: polled continuously, some samples never
// show all their ready bits, and a channel's data can change before its bit
// is set. So a new sample is also recognised by its data changing (only a
// sample identical to the previous one on all three channels, e.g. every
// channel saturated, then depends on the ready bits). The channels of a
// module update at different times, so the sample is read STREAM_SETTLE_US
// after it was first seen, and stamped with the time it was seen
// (resolution: one polling round).
void
pollStreaming()
{
    uint8_t status_cmd = ADS_DATA_STATUS_REG | READ;
    uint8_t data_cmd = ADS_DATA_CH1_ECG_REG | READ;
    uint8_t status;
    uint8_t ecg[9];
    for(int m = 0; m < clvHd.nbModules(); m++)
    {
        if(!(stream_mask & ((uint32_t)1 << m)))
            continue;
        uint32_t t = micros();
        if(stream_pending[m])
        {
            if(t - stream_seen_us[m] < STREAM_SETTLE_US)
                continue;
            clvHd.readCmd(1, &data_cmd, 9, stream_last[m], m + 1);
            stream_pending[m] = false;
            sendFrame(FRAME_DATA, m, stream_count[m]++, stream_seen_us[m], stream_status[m],
                      stream_last[m]);
            continue;
        }
        clvHd.readCmd(1, &status_cmd, 1, &status, m + 1);
        clvHd.readCmd(1, &data_cmd, 9, ecg, m + 1);
        if((status & stream_drdy) == stream_drdy || memcmp(ecg, stream_last[m], 9) != 0)
        {
            stream_pending[m] = true;
            stream_seen_us[m] = t;
            stream_status[m] = status;
        }
    }
}

void
setup()
{
    Serial.begin(460800);
    delay(1000);
    clvHd.begin();
}

void
loop()
{
    if(streaming)
    {
        // only 'x' (stop) and 'i' (LED) are accepted while streaming
        while(Serial.available() >= 1)
        {
            int c = Serial.read();
            if(c == 'x')
            {
                stopStreaming();
                return;
            }
            if(c == 'i')
                setLed();
        }
        pollStreaming();
        return;
    }

    if(Serial.available() >= 1)
    {
        recv_buff[0] = Serial.read();
        switch(recv_buff[0])
        {
        case 'r': // Reading cmd > 'r' | mask_id | nb_bytes_to_read | n_cmd | cmd[n_cmd] : read n bytes starting from reg
        {
            Serial.readBytes((char *)recv_buff + 1, 6);
            mask_id = *((uint32_t *)(recv_buff + 1)); //4 bytes mask_id
            n = recv_buff[5];     //1 byte number of bytes to read
            n_cmd = recv_buff[6]; //1 byte size of the command
            if(n_cmd > (int)RECV_PAYLOAD_MAX)
                n_cmd = 0; // malformed: reply with no data below
            Serial.readBytes((char *)recv_buff + 7, n_cmd);
            *timestamp = micros(); //8 bytes timestamp stored in send_buff
            int ir = 0;
            for(i = 0; i < clvHd.nbModules() && n_cmd > 0; i++)
            {
                if(mask_id & ((uint32_t)1 << i)) //check if the i-th bit is set
                {
                    if(n * (ir + 1) > 255 || n * (ir + 1) > (int)SEND_PAYLOAD_MAX)
                        break; // the reply length is a single byte
                    //read n bytes starting from reg address of the module i
                    //and store them in vals_buff (send_buff + 9)
                    clvHd.readCmd(n_cmd, recv_buff + 7, n, vals_buff + n * ir,
                                  i + 1);
                    ir++;
                }
            }
            *size_buff = n * ir; //number of bytes read (send_buff + 8)
            Serial.write(send_buff, 9 + (*size_buff));
            break;
        }
        case 'w': //> 'w' | mask id | n | n_cmd | cmd[n_cmd] | val[n] : write n bytes starting from reg
        {
            Serial.readBytes((char *)recv_buff + 1, 6);
            mask_id = *((uint32_t *)(recv_buff + 1)); //4 bytes mask_id
            n = recv_buff[5];     //1 byte number of bytes to write
            n_cmd = recv_buff[6]; //1 byte size of the command
            if(n_cmd + n > (int)RECV_PAYLOAD_MAX)
            {
                // malformed: drain the payload and write nothing
                for(int k = 0; k < n_cmd + n; k++) Serial.read();
                break;
            }
            Serial.readBytes((char *)recv_buff + 7,
                             n_cmd + n); //read n_cmd + n bytes
            *timestamp = micros();       //8 bytes timestamp stored in send_buff
            for(i = 0; i < clvHd.nbModules(); i++)
            {
                if(mask_id & ((uint32_t)1 << i)) //check if the i-th bit is set
                {
                    //write n bytes starting from reg address of the module i
                    clvHd.writeCmd(n_cmd, recv_buff + 7, n,
                                   recv_buff + 7 + n_cmd, i + 1);
                }
            }
            break;
        }
        case 'n': // Nb module cmd > 'n'
        {
            *timestamp = micros();
            vals_buff[0] = clvHd.nbModules();
            *size_buff = 1;
            Serial.write(send_buff, 9 + *size_buff);
            break;
        }
        case 's': // Setup cmd > 's' : address the modules, reply nb | status
        {
            uint8_t status = CLVHD_INIT_OK;
            uint8_t nb = clvHd.initModules(&status);
            *timestamp = micros();
            vals_buff[0] = nb;
            vals_buff[1] = status;
            *size_buff = 2;
            Serial.write(send_buff, 9 + *size_buff);
            break;
        }
        case 'i': // I2c cmd > 'i' | id | led | r | g | b
        {
            setLed();
            break;
        }
        case 'm': // Mirror cmd > 'm' | b1 | b2 | b3
        {
            Serial.readBytes((char *)recv_buff + 1, 3);
            *timestamp = micros();
            vals_buff[0] = recv_buff[1];
            vals_buff[1] = recv_buff[2];
            vals_buff[2] = recv_buff[3];
            *size_buff = 3;
            Serial.write(send_buff, 9 + *size_buff);
            break;
        }
        case 'v': // Version cmd > 'v'
        {
            *timestamp = micros();
            vals_buff[0] = VERSION_MAJOR;
            vals_buff[1] = VERSION_MINOR;
            *size_buff = 2;
            Serial.write(send_buff, 9 + *size_buff);
            break;
        }
        case 'c': // SPI clock cmd > 'c' | hz(4) : reply the clock set (4)
        {
            Serial.readBytes((char *)recv_buff + 1, 4);
            uint32_t hz = clvHd.setSpiClock(*((uint32_t *)(recv_buff + 1)));
            *timestamp = micros();
            memcpy(vals_buff, &hz, 4);
            *size_buff = 4;
            Serial.write(send_buff, 9 + *size_buff);
            break;
        }
        case 'A': // Stream cmd > 'A' | mask_id(4) | drdy_mask(1) : start conversions and stream frames until 'x'
        {
            Serial.readBytes((char *)recv_buff + 1, 5);
            stream_mask = *((uint32_t *)(recv_buff + 1));
            stream_drdy = recv_buff[5];
            if(stream_drdy == 0)
                stream_drdy = 0xE0; // all three ECG channels
            startStreaming();
            break;
        }
        case 'x': // Stop streaming: nothing to do (and no reply) when not streaming,
        {         // so a host sending it on connect gets no stray bytes
            break;
        }
        }
    }
}
