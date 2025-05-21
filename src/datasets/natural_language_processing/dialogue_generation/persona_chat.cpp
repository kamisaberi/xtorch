#include "datasets/natural_language_processing/dialogue_generation/persona_chat.h"

namespace xt::data::datasets
{
    // ---------------------- PersonaChat ---------------------- //

    PersonaChat::PersonaChat(const std::string& root): PersonaChat::PersonaChat(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    PersonaChat::PersonaChat(const std::string& root, xt::datasets::DataMode mode): PersonaChat::PersonaChat(
        root, mode, false, nullptr, nullptr)
    {
    }

    PersonaChat::PersonaChat(const std::string& root, xt::datasets::DataMode mode, bool download) :
        PersonaChat::PersonaChat(
            root, mode, download, nullptr, nullptr)
    {
    }

    PersonaChat::PersonaChat(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : PersonaChat::PersonaChat(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    PersonaChat::PersonaChat(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void PersonaChat::load_data()
    {

    }

    void PersonaChat::check_resources()
    {

    }
}
