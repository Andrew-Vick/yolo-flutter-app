import 'dart:async';
import 'dart:ui' as ui;
import 'dart:io' as io;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:video_player/video_player.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/yolo_model.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

void main() => runApp(VideoPlayerApp());

class VideoPlayerApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Video Player',
      home: VideoPlayerScreen(),
    );
  }
}

class VideoPlayerScreen extends StatefulWidget {
  @override
  _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late VideoPlayerController _controller;
  late Future<void> _initializeVideoPlayerFuture;
  final GlobalKey _videoKey = GlobalKey(); // For capturing frames
  final List<String> _frameCache = [];
  bool _isProcessing = false;
  late ObjectDetector _predictor;

  @override
  void initState() {
    super.initState();

    _controller = VideoPlayerController.asset('assets/IMG_7932.MOV');

    // Assign _initializeVideoPlayerFuture before using it
    _initializeVideoPlayerFuture = _controller.initialize().then((_) {
      setState(() {}); // Update UI once initialized
      _controller.setLooping(true);
      _startFrameExtraction(); // Start processing frames after initialization
    });
  }


  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  /// Starts extracting frames periodically
  void _startFrameExtraction() async {
    _predictor = await _initObjectDetectorWithLocalModel();
    await _predictor.loadModel(useGpu: true);

    Timer.periodic(Duration(milliseconds: 100), (timer) async {
      if (!_controller.value.isPlaying) {
        timer.cancel();
        return;
      }

      final framePath = await _captureFrame();
      if (framePath != null) {
        _frameCache.add(framePath);
        _processFrame(framePath);
      }
    });
  }

  /// Extract a frame using RenderRepaintBoundary
  Future<String?> _captureFrame() async {
    try {
      RenderRepaintBoundary? boundary = _videoKey.currentContext
          ?.findRenderObject() as RenderRepaintBoundary?;

      if (boundary == null) {
        print("Failed to capture frame: boundary is null");
        return null;
      }

      ui.Image image = await boundary.toImage();
      ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);
      if (byteData != null) {
        Uint8List imageBytes = byteData.buffer.asUint8List();
        String tempDir = (await getTemporaryDirectory()).path;
        String filePath =
            '$tempDir/frame_${DateTime.now().millisecondsSinceEpoch}.png';

        io.File file = io.File(filePath);
        await file.writeAsBytes(imageBytes);
        return file.path;
      }
    } catch (e) {
      print("Error capturing frame: $e");
    }
    return null;
  }

  /// Runs YOLO detection on a frame
  void _processFrame(String framePath) async {
    print("Processing frame: $framePath");
    try {
      print("Starting detection...");
      final List<DetectedObject?>? rawResult =
          await _predictor.detect(imagePath: framePath);

      print("Raw detection result: $rawResult");

      if (rawResult == null || rawResult.isEmpty) {
        print("No objects detected.");
        return;
      }

      // No need for further conversion—directly filter out nulls
      List<DetectedObject> detections =
          rawResult.whereType<DetectedObject>().toList();

      print("Detections found: ${detections.length}");

      setState(() {
        _drawBoundingBoxes(detections);
      });
    } catch (e, stacktrace) {
      print("Error processing frame: $e");
      print(stacktrace);
    }
  }







  Widget _drawBoundingBoxes(List<DetectedObject?>? detections) {
    if (detections == null || detections.isEmpty) {
      return Stack(); // Return an empty widget if there's no detection
    }

    return Stack(
      children: detections
          .where((detection) => detection != null) // Remove nulls
          .map((detection) => Positioned(
                left: detection!.boundingBox.left, // Safe since nulls are filtered out
                top: detection.boundingBox.top,
                width: detection.boundingBox.width,
                height: detection.boundingBox.height,
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.red, width: 2),
                  ),
                  child: Text(
                    detection.label,
                    style: TextStyle(
                        color: Colors.white, backgroundColor: Colors.red),
                  ),
                ),
              ))
          .toList(),
    );
  }


  Future<ObjectDetector> _initObjectDetectorWithLocalModel() async {
    final modelPath = await _copy('assets/best.mlmodel');
    final model = LocalYoloModel(
      id: '',
      task: Task.detect,
      format: Format.coreml,
      modelPath: modelPath,
    );

    return ObjectDetector(model: model);
  }

  Future<String> _copy(String assetPath) async {
    final path = '${(await getApplicationSupportDirectory()).path}/$assetPath';
    await io.Directory(dirname(path)).create(recursive: true);
    final file = io.File(path);
    if (!await file.exists()) {
      final byteData = await rootBundle.load(assetPath);
      await file.writeAsBytes(byteData.buffer
          .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }
    return file.path;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLO Video Player'),
      ),
      body: Stack(
        children: [
          RepaintBoundary(
            key: _videoKey, // Needed for frame extraction
            child: FutureBuilder(
              future: _initializeVideoPlayerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return AspectRatio(
                    aspectRatio: _controller.value.aspectRatio,
                    child: VideoPlayer(_controller),
                  );
                } else if (snapshot.hasError) {
                  return Center(
                    child: Text('Error loading video: ${snapshot.error}'),
                  );
                } else {
                  return const Center(
                    child: CircularProgressIndicator(),
                  );
                }
              },
            ),
          ),
          _drawBoundingBoxes([]), // Placeholder for bounding boxes
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            if (_controller.value.isPlaying) {
              _controller.pause();
            } else {
              _controller.play();
              _startFrameExtraction();
            }
          });
        },
        child: Icon(
          _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
        ),
      ),
    );
  }
}
