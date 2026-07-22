// MESSAGE TARGET_POINT support class

#pragma once

namespace mavlink {
namespace myformat {
namespace msg {

/**
 * @brief TARGET_POINT message
 *
 * Target Point Format: X, Y
 */
struct TARGET_POINT : mavlink::Message {
    static constexpr msgid_t MSG_ID = 42000;
    static constexpr size_t LENGTH = 8;
    static constexpr size_t MIN_LENGTH = 8;
    static constexpr uint8_t CRC_EXTRA = 138;
    static constexpr auto NAME = "TARGET_POINT";


    float x; /*<  X coordinate */
    float y; /*<  Y coordinate */


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

        return ss.str();
    }

    inline void serialize(mavlink::MsgMap &map) const override
    {
        map.reset(MSG_ID, LENGTH);

        map << x;                             // offset: 0
        map << y;                             // offset: 4
    }

    inline void deserialize(mavlink::MsgMap &map) override
    {
        map >> x;                             // offset: 0
        map >> y;                             // offset: 4
    }
};

} // namespace msg
} // namespace myformat
} // namespace mavlink
