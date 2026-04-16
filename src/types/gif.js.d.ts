declare module "gif.js" {
  interface GIFOptions {
    workers?: number;
    quality?: number;
    width?: number;
    height?: number;
    workerScript?: string;
    background?: string;
    repeat?: number;
    dither?: boolean | string;
    transparent?: string | null;
    debug?: boolean;
  }

  interface FrameOptions {
    delay?: number;
    copy?: boolean;
    dispose?: number;
  }

  type FrameSource =
    | HTMLCanvasElement
    | CanvasRenderingContext2D
    | HTMLImageElement;

  class GIF {
    constructor(options?: GIFOptions);
    addFrame(source: FrameSource, options?: FrameOptions): void;
    on(event: "start", callback: () => void): void;
    on(event: "abort", callback: () => void): void;
    on(event: "progress", callback: (progress: number) => void): void;
    on(event: "finished", callback: (blob: Blob) => void): void;
    render(): void;
    abort(): void;
  }

  export default GIF;
}
