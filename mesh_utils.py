from firedrake import (
    Mesh,
    RectangleMesh,
    UnitDiskMesh,
    MeshHierarchy,
    PETSc,
)


def make_mesh(opts):
    meshfile = opts.getString("meshfile", "")
    refine = opts.getInt("refine", 0)
    bary = opts.getBool("bary", False)
    tohex = opts.getBool("tohex", False)
    if meshfile != "":
        msh = Mesh(meshfile)
    else:
        meshtype = opts.getString("meshtype", "rectangle")
        if meshtype == "rectangle":
            quad = opts.getBool("quad", False)
            nx = opts.getInt("nx", 4)
            ny = opts.getInt("ny", nx)
            urx = opts.getReal("urx", 1.0)
            ury = opts.getReal("ury", urx)
            llx = opts.getReal("llx", 0.0)
            lly = opts.getReal("lly", 0.0)
            diagonal = opts.getString("mesh_diagonal", "left") if not quad else None
            msh = RectangleMesh(
                nx,
                ny,
                urx,
                ury,
                quadrilateral=quad,
                originX=llx,
                originY=lly,
                diagonal=diagonal,
            )
        elif meshtype == "disk":
            ref_lev = opts.getInt("disk_ref_level", 4)
            msh = UnitDiskMesh(refinement_level=ref_lev)
    msh = MeshHierarchy(msh, refine)[-1]
    if bary:
        dm = msh._topology_dm
        tr = PETSc.DMPlexTransform().create(comm=dm.getComm())
        tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
        tr.setDM(dm)
        tr.setUp()
        rdm = tr.apply(dm)
        msh = Mesh(rdm)
    if tohex:
        dm = msh._topology_dm
        tr = PETSc.DMPlexTransform().create(comm=dm.getComm())
        tr.setType(PETSc.DMPlexTransformType.REFINETOBOX)
        tr.setDM(dm)
        tr.setUp()
        rdm = tr.apply(dm)
        msh = Mesh(rdm)
    msh._topology_dm.viewFromOptions("-mesh_view")

    return msh
