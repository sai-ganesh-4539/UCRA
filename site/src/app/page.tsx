import { Nav } from "@/components/ucra/nav";
import { Hero } from "@/components/ucra/hero";
import { Problem } from "@/components/ucra/problem";
import { Results } from "@/components/ucra/results";
import { KappaLab } from "@/components/ucra/kappa-lab";
import { DriftDemo } from "@/components/ucra/drift-demo";
import { Cttc } from "@/components/ucra/cttc";
import { Pipeline } from "@/components/ucra/pipeline";
import { Gallery } from "@/components/ucra/gallery";
import { Repro, Footer } from "@/components/ucra/repro";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-[#070b09] text-foreground">
      <Nav />
      <main className="flex-1">
        <Hero />
        <Problem />
        <Results />
        <KappaLab />
        <DriftDemo />
        <Cttc />
        <Pipeline />
        <Gallery />
        <Repro />
      </main>
      <Footer />
    </div>
  );
}
