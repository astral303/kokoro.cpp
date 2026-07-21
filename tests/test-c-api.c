#include "kokoro.h"

#include <stddef.h>
#include <string.h>

int main(void) {
    struct kokoro_context_params params = kokoro_context_default_params();
    if (params.abi_version != KOKORO_ABI_VERSION) {
        return 1;
    }
    if (strcmp(params.espeak_lang, "en-us") != 0) {
        return 2;
    }
    if (kokoro_version() == NULL || kokoro_last_error() == NULL) {
        return 3;
    }

    struct kokoro_audio audio = {0};
    kokoro_audio_free(&audio);
    return audio.samples == NULL && audio.n_samples == 0 ? 0 : 4;
}
