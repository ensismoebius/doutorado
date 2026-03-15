#include <gtest/gtest.h>

#include <stdexcept>

#include "nn/dataLoaders/10.1117/InputModeCodec.hpp"

TEST(InputModeCodecTest, EnumToTokenUsesCanonicalNames)
{
    EXPECT_EQ(protocol101117InputModeToToken(Protocol101117InputMode::Concatenated),
              "concatenated");
    EXPECT_EQ(protocol101117InputModeToToken(Protocol101117InputMode::EegOnly), "eeg-only");
    EXPECT_EQ(protocol101117InputModeToToken(Protocol101117InputMode::AudioOnly), "audio-only");
}

TEST(InputModeCodecTest, ParseTokenIsCaseInsensitive)
{
    EXPECT_EQ(parseProtocol101117InputModeToken("CONCATENATED"),
              Protocol101117InputMode::Concatenated);
    EXPECT_EQ(parseProtocol101117InputModeToken("Eeg-Only"), Protocol101117InputMode::EegOnly);
    EXPECT_EQ(parseProtocol101117InputModeToken("audio-only"), Protocol101117InputMode::AudioOnly);
}

TEST(InputModeCodecTest, SupportedTokensContainAllModes)
{
    const auto tokens = supportedProtocol101117InputModeTokens();
    ASSERT_EQ(tokens.size(), 3U);
    EXPECT_EQ(tokens[0], "concatenated");
    EXPECT_EQ(tokens[1], "eeg-only");
    EXPECT_EQ(tokens[2], "audio-only");
}

TEST(InputModeCodecTest, UnknownTokenThrows)
{
    EXPECT_THROW((void) parseProtocol101117InputModeToken("other"), std::runtime_error);
}
