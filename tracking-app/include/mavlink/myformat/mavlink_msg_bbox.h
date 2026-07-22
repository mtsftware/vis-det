#pragma once
// MESSAGE BBOX PACKING

#define MAVLINK_MSG_ID_BBOX 42001


typedef struct __mavlink_bbox_t {
 float x; /*<  X coordinate*/
 float y; /*<  Y coordinate*/
 float w; /*<  Width*/
 float h; /*<  Height*/
} mavlink_bbox_t;

#define MAVLINK_MSG_ID_BBOX_LEN 16
#define MAVLINK_MSG_ID_BBOX_MIN_LEN 16
#define MAVLINK_MSG_ID_42001_LEN 16
#define MAVLINK_MSG_ID_42001_MIN_LEN 16

#define MAVLINK_MSG_ID_BBOX_CRC 251
#define MAVLINK_MSG_ID_42001_CRC 251



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_BBOX { \
    42001, \
    "BBOX", \
    4, \
    {  { "x", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_bbox_t, x) }, \
         { "y", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_bbox_t, y) }, \
         { "w", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_bbox_t, w) }, \
         { "h", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_bbox_t, h) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_BBOX { \
    "BBOX", \
    4, \
    {  { "x", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_bbox_t, x) }, \
         { "y", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_bbox_t, y) }, \
         { "w", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_bbox_t, w) }, \
         { "h", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_bbox_t, h) }, \
         } \
}
#endif

/**
 * @brief Pack a bbox message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param x  X coordinate
 * @param y  Y coordinate
 * @param w  Width
 * @param h  Height
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_bbox_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float x, float y, float w, float h)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_BBOX_LEN];
    _mav_put_float(buf, 0, x);
    _mav_put_float(buf, 4, y);
    _mav_put_float(buf, 8, w);
    _mav_put_float(buf, 12, h);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_BBOX_LEN);
#else
    mavlink_bbox_t packet;
    packet.x = x;
    packet.y = y;
    packet.w = w;
    packet.h = h;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_BBOX_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_BBOX;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
}

/**
 * @brief Pack a bbox message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param status MAVLink status structure
 * @param msg The MAVLink message to compress the data into
 *
 * @param x  X coordinate
 * @param y  Y coordinate
 * @param w  Width
 * @param h  Height
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_bbox_pack_status(uint8_t system_id, uint8_t component_id, mavlink_status_t *_status, mavlink_message_t* msg,
                               float x, float y, float w, float h)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_BBOX_LEN];
    _mav_put_float(buf, 0, x);
    _mav_put_float(buf, 4, y);
    _mav_put_float(buf, 8, w);
    _mav_put_float(buf, 12, h);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_BBOX_LEN);
#else
    mavlink_bbox_t packet;
    packet.x = x;
    packet.y = y;
    packet.w = w;
    packet.h = h;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_BBOX_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_BBOX;
#if MAVLINK_CRC_EXTRA
    return mavlink_finalize_message_buffer(msg, system_id, component_id, _status, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
#else
    return mavlink_finalize_message_buffer(msg, system_id, component_id, _status, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN);
#endif
}

/**
 * @brief Pack a bbox message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param x  X coordinate
 * @param y  Y coordinate
 * @param w  Width
 * @param h  Height
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_bbox_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float x,float y,float w,float h)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_BBOX_LEN];
    _mav_put_float(buf, 0, x);
    _mav_put_float(buf, 4, y);
    _mav_put_float(buf, 8, w);
    _mav_put_float(buf, 12, h);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_BBOX_LEN);
#else
    mavlink_bbox_t packet;
    packet.x = x;
    packet.y = y;
    packet.w = w;
    packet.h = h;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_BBOX_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_BBOX;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
}

/**
 * @brief Encode a bbox struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param bbox C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_bbox_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_bbox_t* bbox)
{
    return mavlink_msg_bbox_pack(system_id, component_id, msg, bbox->x, bbox->y, bbox->w, bbox->h);
}

/**
 * @brief Encode a bbox struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param bbox C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_bbox_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_bbox_t* bbox)
{
    return mavlink_msg_bbox_pack_chan(system_id, component_id, chan, msg, bbox->x, bbox->y, bbox->w, bbox->h);
}

/**
 * @brief Encode a bbox struct with provided status structure
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param status MAVLink status structure
 * @param msg The MAVLink message to compress the data into
 * @param bbox C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_bbox_encode_status(uint8_t system_id, uint8_t component_id, mavlink_status_t* _status, mavlink_message_t* msg, const mavlink_bbox_t* bbox)
{
    return mavlink_msg_bbox_pack_status(system_id, component_id, _status, msg,  bbox->x, bbox->y, bbox->w, bbox->h);
}

/**
 * @brief Send a bbox message
 * @param chan MAVLink channel to send the message
 *
 * @param x  X coordinate
 * @param y  Y coordinate
 * @param w  Width
 * @param h  Height
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_bbox_send(mavlink_channel_t chan, float x, float y, float w, float h)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_BBOX_LEN];
    _mav_put_float(buf, 0, x);
    _mav_put_float(buf, 4, y);
    _mav_put_float(buf, 8, w);
    _mav_put_float(buf, 12, h);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_BBOX, buf, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
#else
    mavlink_bbox_t packet;
    packet.x = x;
    packet.y = y;
    packet.w = w;
    packet.h = h;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_BBOX, (const char *)&packet, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
#endif
}

/**
 * @brief Send a bbox message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_bbox_send_struct(mavlink_channel_t chan, const mavlink_bbox_t* bbox)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_bbox_send(chan, bbox->x, bbox->y, bbox->w, bbox->h);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_BBOX, (const char *)bbox, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
#endif
}

#if MAVLINK_MSG_ID_BBOX_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This variant of _send() can be used to save stack space by reusing
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_bbox_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float x, float y, float w, float h)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, x);
    _mav_put_float(buf, 4, y);
    _mav_put_float(buf, 8, w);
    _mav_put_float(buf, 12, h);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_BBOX, buf, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
#else
    mavlink_bbox_t *packet = (mavlink_bbox_t *)msgbuf;
    packet->x = x;
    packet->y = y;
    packet->w = w;
    packet->h = h;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_BBOX, (const char *)packet, MAVLINK_MSG_ID_BBOX_MIN_LEN, MAVLINK_MSG_ID_BBOX_LEN, MAVLINK_MSG_ID_BBOX_CRC);
#endif
}
#endif

#endif

// MESSAGE BBOX UNPACKING


/**
 * @brief Get field x from bbox message
 *
 * @return  X coordinate
 */
static inline float mavlink_msg_bbox_get_x(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  0);
}

/**
 * @brief Get field y from bbox message
 *
 * @return  Y coordinate
 */
static inline float mavlink_msg_bbox_get_y(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  4);
}

/**
 * @brief Get field w from bbox message
 *
 * @return  Width
 */
static inline float mavlink_msg_bbox_get_w(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field h from bbox message
 *
 * @return  Height
 */
static inline float mavlink_msg_bbox_get_h(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Decode a bbox message into a struct
 *
 * @param msg The message to decode
 * @param bbox C-struct to decode the message contents into
 */
static inline void mavlink_msg_bbox_decode(const mavlink_message_t* msg, mavlink_bbox_t* bbox)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    bbox->x = mavlink_msg_bbox_get_x(msg);
    bbox->y = mavlink_msg_bbox_get_y(msg);
    bbox->w = mavlink_msg_bbox_get_w(msg);
    bbox->h = mavlink_msg_bbox_get_h(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_BBOX_LEN? msg->len : MAVLINK_MSG_ID_BBOX_LEN;
        memset(bbox, 0, MAVLINK_MSG_ID_BBOX_LEN);
    memcpy(bbox, _MAV_PAYLOAD(msg), len);
#endif
}
