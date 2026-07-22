/** @file
 *	@brief MAVLink comm protocol generated from myformat.xml
 *	@see http://mavlink.org
 */

#pragma once

#include <array>
#include <cstdint>
#include <sstream>

#ifndef MAVLINK_STX
#define MAVLINK_STX 253
#endif

#include "../message.hpp"

namespace mavlink {
namespace myformat {

/**
 * Array of msg_entry needed for @p mavlink_parse_char() (through @p mavlink_get_msg_entry())
 */
constexpr std::array<mavlink_msg_entry_t, 2> MESSAGE_ENTRIES {{ {42000, 138, 8, 8, 0, 0, 0}, {42001, 251, 16, 16, 0, 0, 0} }};

//! MAVLINK VERSION
constexpr auto MAVLINK_VERSION = 3;


// ENUM DEFINITIONS




} // namespace myformat
} // namespace mavlink

// MESSAGE DEFINITIONS
#include "./mavlink_msg_target_point.hpp"
#include "./mavlink_msg_bbox.hpp"

// base include

