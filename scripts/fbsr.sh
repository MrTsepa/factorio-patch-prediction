#!/usr/bin/env bash
# Run FBSR's CLI (com.demod.fbsr.FBSRMain) reproducibly on Apple Silicon.
#
# FBSR (github.com/demodude4u/Factorio-FBSR) is the engine FactorioBin / the Reddit
# BlueprintBot use to render blueprint strings to images. This wrapper builds a
# correct classpath (incl. the arm64 webp-imageio fork + kotlin-stdlib) and runs it.
#
# One-time prerequisites (see docs/findings.md "Game-accurate rendering with FBSR"):
#   - JDK 21 + Maven         (brew install openjdk@21 maven)
#   - the 3 repos built into ~/.m2:
#       demodude4u/Java-Factorio-Data-Wrapper, demodude4u/Discord-Core-Bot-Apple,
#       demodude4u/Factorio-FBSR   (mvn install / mvn package)
#   - a one-time data+sprite bake from your Factorio install:
#       scripts/fbsr.sh cfg-factorio -install=<factorio>/Contents -auto-find-exec
#       (set config.json factorio.executable to the RELATIVE 'MacOS/factorio')
#       scripts/fbsr.sh   then:  profile-default-vanilla -f ; build -a
set -euo pipefail
FBSR_HOME="${FBSR_HOME:-$HOME/Workspace/Factorio-FBSR/FactorioBlueprintStringRenderer}"
export JAVA_HOME="${JAVA_HOME:-$(/opt/homebrew/bin/brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home}"
export PATH="$JAVA_HOME/bin:/opt/homebrew/bin:$PATH"

WEBP_VER="${WEBP_VER:-0.11.0}"        # arm64 macOS webp natives (FBSR ships x86_64-only sejda)
KOTLIN_VER="${KOTLIN_VER:-2.0.21}"    # the usefulness fork is Kotlin
M2="$HOME/.m2/repository"
WEBP="$M2/com/github/usefulness/webp-imageio/$WEBP_VER/webp-imageio-$WEBP_VER.jar"
KOTLIN="$M2/org/jetbrains/kotlin/kotlin-stdlib/$KOTLIN_VER/kotlin-stdlib-$KOTLIN_VER.jar"
SEJDA="$M2/org/sejda/imageio/webp-imageio/0.1.6/webp-imageio-0.1.6.jar"

cd "$FBSR_HOME"
CPF="$FBSR_HOME/.fbsr_cp.txt"
if [ ! -f "$CPF" ]; then
  [ -f "$WEBP" ]   || mvn -q dependency:get -Dartifact="com.github.usefulness:webp-imageio:$WEBP_VER"
  [ -f "$KOTLIN" ] || mvn -q dependency:get -Dartifact="org.jetbrains.kotlin:kotlin-stdlib:$KOTLIN_VER"
  mvn -q dependency:build-classpath -Dmdep.outputFile=.deps_cp
  JAR="$(ls target/FactorioBlueprintStringRenderer-*.jar | head -1)"
  DEPS="$(tr ':' '\n' < .deps_cp | grep -vF "$SEJDA" | paste -sd: -)"   # drop x86-only sejda
  echo "$WEBP:$JAR:$DEPS:$KOTLIN:$FBSR_HOME/lib/*" > "$CPF"
fi
exec java -cp "$(cat "$CPF")" com.demod.fbsr.FBSRMain "$@"
