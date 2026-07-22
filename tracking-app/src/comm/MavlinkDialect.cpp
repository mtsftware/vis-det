// mavlink_helpers.h (MAVLINK_GET_MSG_ENTRY tanımlı) mavlink::mavlink_get_msg_entry()'i
// dışarıdan sağlanmasını bekler (message.hpp §"user of MAVLink library should
// provide implementation"). myformat dialect'i yalnızca MESSAGE_ENTRIES
// dizisini üretiyor (myformat.hpp) — CRC_EXTRA/uzunluk doğrulaması için
// msgid -> entry aramasını burada tek bir çeviri biriminde tanımlıyoruz.
#include "mavlink/myformat/myformat.hpp"

namespace mavlink {

const mavlink_msg_entry_t *mavlink_get_msg_entry(uint32_t msgid) {
    for (const auto &entry : myformat::MESSAGE_ENTRIES) {
        if (entry.msgid == msgid) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace mavlink