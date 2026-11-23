"""Score the extra SMPL-X archive with the unchanged common silhouette worker."""
from capture_processing.tools import evaluate_silhouette_fit as worker


def main():
    original = worker.mesh_provider
    def provider(subject, request):
        if request['kind'] != 'smplx_confidence':
            raise ValueError('Expected the confidence-filtered SMPL-X fit')
        return original(subject, dict(request, kind='smplx'))
    worker.mesh_provider = provider
    try:
        worker.main()
    finally:
        worker.mesh_provider = original


if __name__ == '__main__':
    main()
