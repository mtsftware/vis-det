// MESSAGE BBOX support class

#pragma once

namespace mavlink {
namespace myformat {
namespace msg {

/**
 * @brief BBOX message
 *
 * BBOX Format: X, Y, W, H
 */
struct BBOX : mavlink::Message {
    static constexpr msgid_t MSG_ID = 42001;
    static constexpr size_t LENGTH = 16;
    static constexpr size_t MIN_LENGTH = 16;
    static constexpr uint8_t CRC_EXTRA = 251;
    static constexpr auto NAME = "BBOX";


    float x; /*<  X coordinate */
    float y; /*<  Y coordinate */
    float w; /*<  Width */
    float h; /*<  Height */


    inline std::string get_name(void) const override
    {
            return NAME;
    }

    inline Info get_message_info(void) const override
    {
            return { MSG_ID, LENGTH, MIN_LENGTH, CRC_EXTRA };
    }

    inline std::string to_yaml(void) const override
    {
        std::stringstream ss;

        ss << NAME << ":" << std::endl;
        ss << "  x: " << x << std::endl;
        ss << "  y: " << y << std::endl;
        ss << "  w: " << w << std::endl;
        ss << "  h: " << h << std::endl;

        return ss.str();
    }

    inline void serialize(mavlink::MsgMap &map) const override
    {
        map.reset(MSG_ID, LENGTH);

        map << x;                             // offset: 0
        map << y;                             // offset: 4
        map << w;                             // offset: 8
        map << h;                             // offset: 12
    }

    inline void deserialize(mavlink::MsgMap &map) override
    {
        map >> x;                             // offset: 0
        map >> y;                             // offset: 4
        map >> w;                             // offset: 8
        map >> h;                             // offset: 12
    }
};

} // namespace msg
} // namespace myformat
} // namespace mavlink
