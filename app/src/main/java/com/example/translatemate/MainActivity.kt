package com.example.translatemate

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform


class MainActivity : ComponentActivity() {
    private val RECORD_AUDIO_PERMISSION_CODE = 1
    var mediaPlayer: MediaPlayer? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var lyricsText = ""
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.RECORD_AUDIO), RECORD_AUDIO_PERMISSION_CODE)
        }

        setContent {
            var recording = false
            var recording_text by remember { mutableStateOf(false) }

            // Inside MainActivity class
            var mediaRecorder = MediaRecorder()

            // Define a valid file path within your app's cache directory
            val audioFilePath: String = "${externalCacheDir?.absolutePath}/audio.3gp"

            //Starting python interpreter.
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
            }
            //Getting python interpreter instance.
            val py: Python = Python.getInstance()
            //Creating the object of python source file yolo_module.
            val pyo: PyObject = py.getModule("translate")


            Column(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = if (recording_text) "Recording..." else "Press to Record",
                    color = if (recording_text) Color.Red else Color.Black,
                    fontSize = 24.sp,
                    modifier = Modifier.padding(16.dp)
                )

                Button(
                    onClick = {
                        if (recording) {
                            recording = false
                            recording_text = false

                            // Stop recording audio
                            mediaRecorder.stop()
                            mediaRecorder.reset()
                            mediaRecorder.release()
                            mediaRecorder = MediaRecorder()

                            // Play the recorded audio
                            mediaPlayer = MediaPlayer().apply {
                                setDataSource(audioFilePath)
                                prepare()
                                start()
                            }

                            //Calling function detect_and_draw from module.
                            val obj: PyObject = pyo.callAttr("translate_english_audio_to_chinese_text", audioFilePath)

                            // Display lyrics on the screen
                            lyricsText = obj.toString()
                            val clip: ClipData = ClipData.newPlainText("simple text", lyricsText)
                            clipboard.setPrimaryClip(clip)
                        } else {
                            recording = true
                            recording_text = true
                            // Start recording audio
                            mediaRecorder.apply {
                                setAudioSource(MediaRecorder.AudioSource.MIC)
                                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
                                setOutputFile(audioFilePath)
                                prepare()
                                start()
                            }
                            lyricsText = ""
                        }
                    }
                ) {
                    Text(text = if (recording_text) "Stop" else "Start")
                }

                // Display lyrics here when the recording is stopped.
                Text(
                    text = if (recording_text) lyricsText else lyricsText,
                    color = if (recording_text) Color.Red else Color.Black,
                    fontSize = 24.sp,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }



    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RECORD_AUDIO_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission is granted. You can now use audio recording.
            } else {
                // Permission is denied. Handle the situation as needed.
            }
        }
    }

    override fun onStop() {
        super.onStop()
        mediaPlayer?.release()
        mediaPlayer = null
    }
}