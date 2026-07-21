// Utilities for normalizing espeak-ng phoneme output.
#pragma once

#include <string>

namespace kokoro_phon {

// Strip espeak-ng language-switch markers like (en), (it), (de-AT) from
// IPA output.
inline void strip_espeak_lang_markers(std::string& ipa) {
    size_t out = 0;
    size_t len = ipa.size();
    for (size_t i = 0; i < len;) {
        if (ipa[i] == '(' && i + 3 < len) {
            size_t j = i + 1;
            size_t alpha_start = j;
            while (j < len && j - alpha_start < 3 && ipa[j] >= 'a' && ipa[j] <= 'z')
                j++;
            if (j - alpha_start >= 2) {
                size_t before_region = j;
                if (j < len && ipa[j] == '-') {
                    j++;
                    size_t reg_start = j;
                    while (j < len && j - reg_start < 4 &&
                           ((ipa[j] >= 'a' && ipa[j] <= 'z') || (ipa[j] >= 'A' && ipa[j] <= 'Z') ||
                            (ipa[j] >= '0' && ipa[j] <= '9')))
                        j++;
                    if (j - reg_start < 2)
                        j = before_region;
                }
                if (j < len && ipa[j] == ')') {
                    j++;
                    if (j < len && ipa[j] == ' ')
                        j++;
                    i = j;
                    continue;
                }
            }
        }
        ipa[out++] = ipa[i++];
    }
    ipa.resize(out);
}
} // namespace kokoro_phon
