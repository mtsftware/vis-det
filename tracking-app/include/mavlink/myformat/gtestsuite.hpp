/** @file
 *	@brief MAVLink comm testsuite protocol generated from myformat.xml
 *	@see http://mavlink.org
 */

#pragma once

#include <gtest/gtest.h>
#include "myformat.hpp"

#ifdef TEST_INTEROP
using namespace mavlink;
#undef MAVLINK_HELPER
#include "mavlink.h"
#endif


TEST(myformat, TARGET_POINT)
{
    mavlink::mavlink_message_t msg;
    mavlink::MsgMap map1(msg);
    mavlink::MsgMap map2(msg);

    mavlink::myformat::msg::TARGET_POINT packet_in{};
    packet_in.x = 17.0;
    packet_in.y = 45.0;

    mavlink::myformat::msg::TARGET_POINT packet1{};
    mavlink::myformat::msg::TARGET_POINT packet2{};

    packet1 = packet_in;

    //std::cout << packet1.to_yaml() << std::endl;

    packet1.serialize(map1);

    mavlink::mavlink_finalize_message(&msg, 1, 1, packet1.MIN_LENGTH, packet1.LENGTH, packet1.CRC_EXTRA);

    packet2.deserialize(map2);

    EXPECT_EQ(packet1.x, packet2.x);
    EXPECT_EQ(packet1.y, packet2.y);
}

#ifdef TEST_INTEROP
TEST(myformat_interop, TARGET_POINT)
{
    mavlink_message_t msg;

    // to get nice print
    memset(&msg, 0, sizeof(msg));

    mavlink_target_point_t packet_c {
         17.0, 45.0
    };

    mavlink::myformat::msg::TARGET_POINT packet_in{};
    packet_in.x = 17.0;
    packet_in.y = 45.0;

    mavlink::myformat::msg::TARGET_POINT packet2{};

    mavlink_msg_target_point_encode(1, 1, &msg, &packet_c);

    // simulate message-handling callback
    [&packet2](const mavlink_message_t *cmsg) {
        MsgMap map2(cmsg);

        packet2.deserialize(map2);
    } (&msg);

    EXPECT_EQ(packet_in.x, packet2.x);
    EXPECT_EQ(packet_in.y, packet2.y);

#ifdef PRINT_MSG
    PRINT_MSG(msg);
#endif
}
#endif

TEST(myformat, BBOX)
{
    mavlink::mavlink_message_t msg;
    mavlink::MsgMap map1(msg);
    mavlink::MsgMap map2(msg);

    mavlink::myformat::msg::BBOX packet_in{};
    packet_in.x = 17.0;
    packet_in.y = 45.0;
    packet_in.w = 73.0;
    packet_in.h = 101.0;

    mavlink::myformat::msg::BBOX packet1{};
    mavlink::myformat::msg::BBOX packet2{};

    packet1 = packet_in;

    //std::cout << packet1.to_yaml() << std::endl;

    packet1.serialize(map1);

    mavlink::mavlink_finalize_message(&msg, 1, 1, packet1.MIN_LENGTH, packet1.LENGTH, packet1.CRC_EXTRA);

    packet2.deserialize(map2);

    EXPECT_EQ(packet1.x, packet2.x);
    EXPECT_EQ(packet1.y, packet2.y);
    EXPECT_EQ(packet1.w, packet2.w);
    EXPECT_EQ(packet1.h, packet2.h);
}

#ifdef TEST_INTEROP
TEST(myformat_interop, BBOX)
{
    mavlink_message_t msg;

    // to get nice print
    memset(&msg, 0, sizeof(msg));

    mavlink_bbox_t packet_c {
         17.0, 45.0, 73.0, 101.0
    };

    mavlink::myformat::msg::BBOX packet_in{};
    packet_in.x = 17.0;
    packet_in.y = 45.0;
    packet_in.w = 73.0;
    packet_in.h = 101.0;

    mavlink::myformat::msg::BBOX packet2{};

    mavlink_msg_bbox_encode(1, 1, &msg, &packet_c);

    // simulate message-handling callback
    [&packet2](const mavlink_message_t *cmsg) {
        MsgMap map2(cmsg);

        packet2.deserialize(map2);
    } (&msg);

    EXPECT_EQ(packet_in.x, packet2.x);
    EXPECT_EQ(packet_in.y, packet2.y);
    EXPECT_EQ(packet_in.w, packet2.w);
    EXPECT_EQ(packet_in.h, packet2.h);

#ifdef PRINT_MSG
    PRINT_MSG(msg);
#endif
}
#endif
