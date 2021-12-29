plugins {
    kotlin("jvm") version "1.6.10"
    java
    id("org.graalvm.buildtools.native") version "0.9.4"
}

group = "com.kenvix"
version = "1.0-SNAPSHOT"

val pMainClass = "com.kenvix.lablocker.Main"

repositories {
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    implementation(kotlin("stdlib"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.6.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}

nativeBuild {
    mainClass.set(pMainClass)
    debug.set(false) // Determines if debug info should be generated, defaults to false
    verbose.set(true) // Add verbose output, defaults to false
    fallback.set(true) // Sets the fallback mode of native-image, defaults to false
    sharedLibrary.set(false) // Determines if image is a shared library, defaults to false if `java-library` plugin isn't included

    // Advanced options
    // buildArgs.add("") // Passes '-H:Extra' to the native image builder options. This can be used to pass parameters which are not directly supported by this extension
    // jvmArgs.add("") // Passes 'flag' directly to the JVM running the native image builder

    // Development options
    agent.set(true) // Enables the reflection agent. Can be also set on command line using '-Pagent'
    useFatJar.set(true) // Instead of passing each jar individually, builds a fat jar
}