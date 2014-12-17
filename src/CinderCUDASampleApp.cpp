#include "Resources.h"
#include "cinder/app/AppNative.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Fbo.h"
#include "cinder/gl/GlslProg.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

class CinderCUDASampleApp : public ci::app::AppNative {
  public:
	void setup();
	void mouseDown( ci::app::MouseEvent event );
	void update();
	void draw();

	int image_width;
	int image_height;
	int num_texels;
	int num_values;
	int size_tex_data;
	unsigned int *cuda_dest_resource;
	struct cudaGraphicsResource *cuda_tex_result_resource;

	ci::gl::Fbo mFbo;

	void generateCUDAImage() {
		unsigned int* out_data = cuda_dest_resource;
		dim3 block(16, 16, 1); 
		dim3 grid(image_width / block.x, image_height / block.y, 1);
		launch_cudaProcess(grid, block, 0, out_data, image_width);

		cudaArray *texture_ptr;
		cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
		cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);

		int num_texels = image_width * image_height;
		int num_values = num_texels * 4;
		int size_tex_data = sizeof(GLubyte) * num_values;
		cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);
	}
};

void CinderCUDASampleApp::setup()
{
	image_width = 640;
	image_height = 480;

	char argv[] = { "cinderCudaSample" };
	findCudaGLDevice(1, NULL);

	// init GL Buffers
	ci::gl::Fbo::Format format;
	format.setColorInternalFormat(GL_RGBA);
	format.setWrapS(GL_CLAMP_TO_EDGE);
	format.setWrapT(GL_CLAMP_TO_EDGE);
	format.setMinFilter(GL_NEAREST);
	format.setMagFilter(GL_NEAREST);
	format.enableMipmapping(false);
	format.enableDepthBuffer(false);
	mFbo = ci::gl::Fbo(image_width, image_height, format);
	cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, mFbo.getId(), GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

	// init CUDA Buffers
	num_texels = image_width * image_height;
	num_values = num_texels * 4;
	size_tex_data = sizeof(GLubyte) * num_values;
	cudaMalloc((void **)&cuda_dest_resource, size_tex_data);
}

void CinderCUDASampleApp::mouseDown( ci::app::MouseEvent event )
{
}

void CinderCUDASampleApp::update()
{
	generateCUDAImage();
	cudaDeviceSynchronize();
}

void CinderCUDASampleApp::draw()
{
	ci::gl::clear( ci::Color( 0, 0, 0 ) ); 
	ci::gl::draw(mFbo.getTexture());
}

CINDER_APP_NATIVE( CinderCUDASampleApp, ci::app::RendererGl )