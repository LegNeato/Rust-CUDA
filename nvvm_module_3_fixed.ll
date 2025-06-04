; ModuleID = 'nvvm_module_3.bc'
source_filename = "0z5jpir3dx5xvvmav265ne935"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@"private$0" = private unnamed_addr constant <{ [27 x i8] }> <{ [27 x i8] c"chunk size must be non-zero" }>, align 1
@"private$1" = private unnamed_addr constant <{ i8*, [8 x i8] }> <{ i8* @"private$0", [8 x i8] c"\1B\00\00\00\00\00\00\00" }>, align 8

declare void @__rust_eh_personality()

; Function Attrs: inlinehint nounwind
define void @"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$16chunks_exact_mut17h8b4674d143cef990E"(i8* noalias nocapture sret({ { i8*, i64 }, { i8*, i64 }, i64, {} }) align 8 dereferenceable(40) %0, i8* nonnull align 1 %1, i64 %2, i64 %3, i8* nonnull align 8 %4) unnamed_addr #0 !dbg !4 {
start:
  %5 = alloca [8 x i8], align 8
  %6 = bitcast ptr %5 to i8*
  %7 = alloca [16 x i8], align 8
  %8 = bitcast ptr %7 to i8*
  %9 = alloca [48 x i8], align 8
  %10 = bitcast ptr %9 to i8*
  %11 = bitcast ptr %8 to i8*
  %bitcast.20 = bitcast ptr %11 to i8**
  store ptr %1, i8** %bitcast.20, align 8
  %12 = getelementptr inbounds i8, ptr %8, i64 8
  %13 = bitcast ptr %12 to i8*
  %bitcast.23 = bitcast ptr %13 to i64*
  store i64 %2, i64* %bitcast.23, align 8
    #dbg_declare(ptr %8, !51, !DIExpression(), !53)
  %14 = bitcast ptr %6 to i8*
  %bitcast.26 = bitcast ptr %14 to i64*
  store i64 %3, i64* %bitcast.26, align 8
    #dbg_declare(ptr %6, !52, !DIExpression(), !54)
  %15 = icmp eq i64 %3, 0, !dbg !55
  br i1 %15, label %bb2, label %bb1, !dbg !55

bb2:                                              ; preds = %start
  %16 = bitcast ptr %10 to i8*, !dbg !56
  call void @_ZN4core3fmt9Arguments9new_const17hdf2f30aca51edfbdE(i8* noalias nocapture align 8 dereferenceable(48) %16, i8* nonnull align 8 @"private$1"), !dbg !56
  %17 = bitcast ptr %10 to i8*, !dbg !56
  call void @_ZN4core9panicking9panic_fmt17hbcd5653c6dc71186E(i8* noalias nocapture align 8 dereferenceable(48) %17, i8* nonnull align 8 %4), !dbg !56
  unreachable, !dbg !56

bb1:                                              ; preds = %start
  call void @"_ZN4core5slice4iter23ChunksExactMut$LT$T$GT$3new17ha5b9700c82ce6b00E"(i8* noalias nocapture align 8 dereferenceable(40) %0, i8* nonnull align 1 %1, i64 %2, i64 %3), !dbg !57
  ret void, !dbg !58
}

; Function Attrs: inlinehint nounwind
define void @"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$22split_at_mut_unchecked17h96cf66b4b6523c29E"(i8* noalias nocapture sret({ { i8*, i64 }, { i8*, i64 } }) align 8 dereferenceable(32) %0, i8* nonnull align 1 %1, i64 %2, i64 %3) unnamed_addr #0 !dbg !59 {
start:
  %4 = alloca [8 x i8], align 8
  %5 = alloca [8 x i8], align 8
  %6 = alloca [16 x i8], align 8
  %7 = alloca [8 x i8], align 8
  %8 = bitcast ptr %7 to i8*
  %9 = alloca [8 x i8], align 8
  %10 = bitcast ptr %9 to i8*
  %11 = alloca [8 x i8], align 8
  %12 = bitcast ptr %11 to i8*
  %13 = alloca [16 x i8], align 8
  %14 = bitcast ptr %13 to i8*
  %15 = bitcast ptr %14 to i8*
  %bitcast.58 = bitcast ptr %15 to i8**
  store ptr %1, i8** %bitcast.58, align 8
  %16 = getelementptr inbounds i8, ptr %14, i64 8
  %17 = bitcast ptr %16 to i8*
  %bitcast.61 = bitcast ptr %17 to i64*
  store i64 %2, i64* %bitcast.61, align 8
    #dbg_declare(ptr %14, !67, !DIExpression(), !74)
  %18 = bitcast ptr %12 to i8*
  %bitcast.64 = bitcast ptr %18 to i64*
  store i64 %3, i64* %bitcast.64, align 8
    #dbg_declare(ptr %12, !68, !DIExpression(), !75)
  %19 = bitcast ptr %10 to i8*, !dbg !76
  %bitcast.67 = bitcast ptr %19 to i64*
  store i64 %2, i64* %bitcast.67, align 8, !dbg !76
    #dbg_declare(ptr %10, !69, !DIExpression(), !77)
  %20 = bitcast ptr %6 to i8*
  %21 = bitcast ptr %20 to i8*
  %bitcast.71 = bitcast ptr %21 to i8**
  store ptr %1, i8** %bitcast.71, align 8
  %22 = getelementptr inbounds i8, ptr %20, i64 8
  %23 = bitcast ptr %22 to i8*
  %bitcast.74 = bitcast ptr %23 to i64*
  store i64 %2, i64* %bitcast.74, align 8
    #dbg_declare(ptr %20, !78, !DIExpression(), !83)
  %24 = bitcast ptr %8 to i8*, !dbg !85
  %bitcast.77 = bitcast ptr %24 to i8**
  store ptr %1, i8** %bitcast.77, align 8, !dbg !85
    #dbg_declare(ptr %8, !71, !DIExpression(), !86)
  br label %bb2, !dbg !87

bb2:                                              ; preds = %start
  call void @"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$22split_at_mut_unchecked18precondition_check17he0bcb88ad1bc973cE"(i64 %3, i64 %2), !dbg !90
  br label %bb3, !dbg !90

bb3:                                              ; preds = %bb2
  %25 = call { i8*, i64 } @_ZN4core5slice3raw18from_raw_parts_mut17h0b15c6eeb412adb8E(ptr %1, i64 %3), !dbg !91
  %26 = extractvalue { i8*, i64 } %25, 0, !dbg !91
  %27 = extractvalue { i8*, i64 } %25, 1, !dbg !91
  %28 = bitcast ptr %4 to i8*
  %29 = bitcast ptr %5 to i8*
  %30 = bitcast ptr %29 to i8*
  %bitcast.92 = bitcast ptr %30 to i8**
  store ptr %1, i8** %bitcast.92, align 8
    #dbg_declare(ptr %29, !92, !DIExpression(), !102)
  %31 = bitcast ptr %28 to i8*
  %bitcast.95 = bitcast ptr %31 to i64*
  store i64 %3, i64* %bitcast.95, align 8
    #dbg_declare(ptr %28, !101, !DIExpression(), !104)
  %32 = call zeroext i1 @_ZN4core9ub_checks17check_language_ub17hfaa6a2ee487d6951E() #2, !dbg !105
  br i1 %32, label %bb2.i, label %"_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add17h1171b2ede0480fe3E.exit", !dbg !105

bb2.i:                                            ; preds = %bb3
  %33 = bitcast ptr %1 to i8*, !dbg !107
  call void @"_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add18precondition_check17hc776c183d9ddbe30E"(ptr %33, i64 %3, i64 1) #2, !dbg !108
  br label %"_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add17h1171b2ede0480fe3E.exit", !dbg !108

"_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add17h1171b2ede0480fe3E.exit": ; preds = %bb2.i, %bb3
  %34 = getelementptr inbounds i8, ptr %1, i64 %3, !dbg !109
  %35 = sub nuw i64 %2, %3, !dbg !110
  %36 = call { i8*, i64 } @_ZN4core5slice3raw18from_raw_parts_mut17h0b15c6eeb412adb8E(ptr %34, i64 %35), !dbg !111
  %37 = extractvalue { i8*, i64 } %36, 0, !dbg !111
  %38 = extractvalue { i8*, i64 } %36, 1, !dbg !111
  %39 = bitcast ptr %0 to i8*, !dbg !112
  store ptr %26, ptr %39, align 8, !dbg !112
  %40 = bitcast ptr %0 to i8*, !dbg !112
  %41 = getelementptr inbounds i8, ptr %40, i64 8, !dbg !112
  %42 = bitcast ptr %41 to i8*, !dbg !112
  store i64 %27, ptr %42, align 8, !dbg !112
  %43 = bitcast ptr %0 to i8*, !dbg !112
  %44 = getelementptr inbounds i8, ptr %43, i64 16, !dbg !112
  %45 = bitcast ptr %44 to i8*, !dbg !112
  store ptr %37, ptr %45, align 8, !dbg !112
  %46 = getelementptr inbounds i8, ptr %44, i64 8, !dbg !112
  %47 = bitcast ptr %46 to i8*, !dbg !112
  store i64 %38, ptr %47, align 8, !dbg !112
  ret void, !dbg !113
}

; Function Attrs: inlinehint nounwind
define { i8*, i8* } @"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$4iter17hffcb055fac345fcdE"(i8* nonnull align 8 %0, i64 %1) unnamed_addr #0 !dbg !114 {
start:
  %2 = alloca [16 x i8], align 8
  %3 = bitcast ptr %2 to i8*
  %4 = bitcast ptr %3 to i8*
  %bitcast.133 = bitcast ptr %4 to i8**
  store ptr %0, i8** %bitcast.133, align 8
  %5 = getelementptr inbounds i8, ptr %3, i64 8
  %6 = bitcast ptr %5 to i8*
  %bitcast.136 = bitcast ptr %6 to i64*
  store i64 %1, i64* %bitcast.136, align 8
    #dbg_declare(ptr %3, !140, !DIExpression(), !141)
  %7 = call { i8*, i8* } @"_ZN4core5slice4iter13Iter$LT$T$GT$3new17h9caec572c36ac217E"(i8* nonnull align 8 %0, i64 %1), !dbg !142
  %8 = extractvalue { i8*, i8* } %7, 0, !dbg !142
  %9 = extractvalue { i8*, i8* } %7, 1, !dbg !142
  %10 = insertvalue { i8*, i8* } undef, ptr %8, 0, !dbg !143
  %11 = insertvalue { i8*, i8* } %10, ptr %9, 1, !dbg !143
  ret { i8*, i8* } %11, !dbg !143
}

; Function Attrs: inlinehint nounwind
declare hidden zeroext i1 @_ZN4core9ub_checks17check_language_ub17hfaa6a2ee487d6951E() unnamed_addr #0

; Function Attrs: inlinehint nounwind
declare hidden void @"_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add18precondition_check17hc776c183d9ddbe30E"(i8*, i64, i64) unnamed_addr #0

; Function Attrs: inlinehint nounwind
declare void @_ZN4core3fmt9Arguments9new_const17hdf2f30aca51edfbdE(i8* noalias nocapture sret({ { i8*, i64 }, { i8*, i64 }, { i8*, i64 } }) align 8 dereferenceable(48), i8* nonnull align 8) unnamed_addr #0

; Function Attrs: inlinehint noreturn nounwind
declare hidden void @_ZN4core9panicking9panic_fmt17hbcd5653c6dc71186E(i8* noalias nocapture align 8 dereferenceable(48), i8* nonnull align 8) unnamed_addr #1

; Function Attrs: inlinehint nounwind
declare void @"_ZN4core5slice4iter23ChunksExactMut$LT$T$GT$3new17ha5b9700c82ce6b00E"(i8* noalias nocapture sret({ { i8*, i64 }, { i8*, i64 }, i64, {} }) align 8 dereferenceable(40), i8* nonnull align 1, i64, i64) unnamed_addr #0

; Function Attrs: inlinehint nounwind
declare hidden void @"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$22split_at_mut_unchecked18precondition_check17he0bcb88ad1bc973cE"(i64, i64) unnamed_addr #0

; Function Attrs: inlinehint nounwind
declare { i8*, i64 } @_ZN4core5slice3raw18from_raw_parts_mut17h0b15c6eeb412adb8E(i8*, i64) unnamed_addr #0

; Function Attrs: inlinehint nounwind
declare { i8*, i8* } @"_ZN4core5slice4iter13Iter$LT$T$GT$3new17h9caec572c36ac217E"(i8* nonnull align 8, i64) unnamed_addr #0

attributes #0 = { inlinehint nounwind }
attributes #1 = { inlinehint noreturn nounwind }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "clang LLVM (rustc_codegen_nvvm)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false)
!1 = !DIFile(filename: "src/lib.rs/@/0z5jpir3dx5xvvmav265ne935", directory: "/root/ed25519-vanity-rs/kernel")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "chunks_exact_mut", linkageName: "_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$16chunks_exact_mut17h8b4674d143cef990E", scope: !6, file: !5, line: 1277, type: !9, scopeLine: 1277, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, templateParams: !35, retainedNodes: !50)
!5 = !DIFile(filename: "/root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/slice/mod.rs", directory: "", checksumkind: CSK_MD5, checksum: "19215b9dc72ad6dba2ae80d466f56756")
!6 = !DINamespace(name: "{impl#0}", scope: !7)
!7 = !DINamespace(name: "slice", scope: !8)
!8 = !DINamespace(name: "core", scope: null)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !24, !22, !37}
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "ChunksExactMut<u8>", scope: !13, file: !12, size: 320, align: 64, flags: DIFlagPublic, elements: !14, templateParams: !35, identifier: "bb9c8086ea5f9b271b48552fd63560f0")
!12 = !DIFile(filename: "<unknown>", directory: "")
!13 = !DINamespace(name: "iter", scope: !7)
!14 = !{!15, !23, !28, !29}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !11, file: !12, baseType: !16, size: 128, align: 64, offset: 128, flags: DIFlagPrivate)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "*mut [u8]", file: !12, size: 128, align: 64, elements: !17, templateParams: !2, identifier: "eb80752d8dc9079cf56e9f0de61d8d5f")
!17 = !{!18, !21}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !16, file: !12, baseType: !19, size: 64, align: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64, align: 64)
!20 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !16, file: !12, baseType: !22, size: 64, align: 64, offset: 64)
!22 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !11, file: !12, baseType: !24, size: 128, align: 64, flags: DIFlagPrivate)
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "&mut [u8]", file: !12, size: 128, align: 64, elements: !25, templateParams: !2, identifier: "bdfeb4840e2373d8742974745efe30b6")
!25 = !{!26, !27}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !24, file: !12, baseType: !19, size: 64, align: 64)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !24, file: !12, baseType: !22, size: 64, align: 64, offset: 64)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "chunk_size", scope: !11, file: !12, baseType: !22, size: 64, align: 64, offset: 256, flags: DIFlagPrivate)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "_marker", scope: !11, file: !12, baseType: !30, align: 8, offset: 320, flags: DIFlagPrivate)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<&mut u8>", scope: !31, file: !12, align: 8, flags: DIFlagPublic, elements: !2, templateParams: !32, identifier: "1b68f41eede940e05c58df69ee1ceb72")
!31 = !DINamespace(name: "marker", scope: !8)
!32 = !{!33}
!33 = !DITemplateTypeParameter(name: "T", type: !34)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut u8", baseType: !20, size: 64, align: 64)
!35 = !{!36}
!36 = !DITemplateTypeParameter(name: "T", type: !20)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&core::panic::location::Location", baseType: !38, size: 64, align: 64)
!38 = !DICompositeType(tag: DW_TAG_structure_type, name: "Location", scope: !39, file: !12, size: 192, align: 64, flags: DIFlagPublic, elements: !41, templateParams: !2, identifier: "a6b02edb6d3bd40ec36f7e395869fe0")
!39 = !DINamespace(name: "location", scope: !40)
!40 = !DINamespace(name: "panic", scope: !8)
!41 = !{!42, !47, !49}
!42 = !DIDerivedType(tag: DW_TAG_member, name: "file", scope: !38, file: !12, baseType: !43, size: 128, align: 64, flags: DIFlagPrivate)
!43 = !DICompositeType(tag: DW_TAG_structure_type, name: "&str", file: !12, size: 128, align: 64, elements: !44, templateParams: !2, identifier: "9277eecd40495f85161460476aacc992")
!44 = !{!45, !46}
!45 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !43, file: !12, baseType: !19, size: 64, align: 64)
!46 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !43, file: !12, baseType: !22, size: 64, align: 64, offset: 64)
!47 = !DIDerivedType(tag: DW_TAG_member, name: "line", scope: !38, file: !12, baseType: !48, size: 32, align: 32, offset: 128, flags: DIFlagPrivate)
!48 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "col", scope: !38, file: !12, baseType: !48, size: 32, align: 32, offset: 160, flags: DIFlagPrivate)
!50 = !{!51, !52}
!51 = !DILocalVariable(name: "self", arg: 1, scope: !4, file: !5, line: 1277, type: !24)
!52 = !DILocalVariable(name: "chunk_size", arg: 2, scope: !4, file: !5, line: 1277, type: !22)
!53 = !DILocation(line: 1277, column: 29, scope: !4)
!54 = !DILocation(line: 1277, column: 40, scope: !4)
!55 = !DILocation(line: 1278, column: 17, scope: !4)
!56 = !DILocation(line: 1278, column: 9, scope: !4)
!57 = !DILocation(line: 1279, column: 9, scope: !4)
!58 = !DILocation(line: 1280, column: 6, scope: !4)
!59 = distinct !DISubprogram(name: "split_at_mut_unchecked", linkageName: "_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$22split_at_mut_unchecked17h96cf66b4b6523c29E", scope: !6, file: !5, line: 2053, type: !60, scopeLine: 2053, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, templateParams: !35, retainedNodes: !66)
!60 = !DISubroutineType(types: !61)
!61 = !{!62, !24, !22}
!62 = !DICompositeType(tag: DW_TAG_structure_type, name: "(&mut [u8], &mut [u8])", file: !12, size: 256, align: 64, elements: !63, templateParams: !2, identifier: "8fb7ceac67daaf55579ad0d4f0caa2df")
!63 = !{!64, !65}
!64 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !62, file: !12, baseType: !24, size: 128, align: 64)
!65 = !DIDerivedType(tag: DW_TAG_member, name: "__1", scope: !62, file: !12, baseType: !24, size: 128, align: 64, offset: 128)
!66 = !{!67, !68, !69, !71}
!67 = !DILocalVariable(name: "self", arg: 1, scope: !59, file: !5, line: 2053, type: !24)
!68 = !DILocalVariable(name: "mid", arg: 2, scope: !59, file: !5, line: 2053, type: !22)
!69 = !DILocalVariable(name: "len", scope: !70, file: !5, line: 2054, type: !22, align: 8)
!70 = distinct !DILexicalBlock(scope: !59, file: !5, line: 2054, column: 9)
!71 = !DILocalVariable(name: "i8*", scope: !72, file: !5, line: 2055, type: !73, align: 8)
!72 = distinct !DILexicalBlock(scope: !70, file: !5, line: 2055, column: 9)
!73 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut u8", baseType: !20, size: 64, align: 64)
!74 = !DILocation(line: 2053, column: 48, scope: !59)
!75 = !DILocation(line: 2053, column: 59, scope: !59)
!76 = !DILocation(line: 2054, column: 19, scope: !59)
!77 = !DILocation(line: 2054, column: 13, scope: !70)
!78 = !DILocalVariable(name: "self", arg: 1, scope: !79, file: !5, line: 771, type: !24)
!79 = distinct !DISubprogram(name: "as_mut_ptr", linkageName: "_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$10as_mut_ptr17hced554ee6a2d7183E", scope: !6, file: !5, line: 771, type: !80, scopeLine: 771, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, templateParams: !35, retainedNodes: !82)
!80 = !DISubroutineType(types: !81)
!81 = !{!73, !24}
!82 = !{!78}
!83 = !DILocation(line: 771, column: 29, scope: !79, inlinedAt: !84)
!84 = distinct !DILocation(line: 2055, column: 19, scope: !70)
!85 = !DILocation(line: 2055, column: 19, scope: !70)
!86 = !DILocation(line: 2055, column: 13, scope: !72)
!87 = !DILocation(line: 74, column: 35, scope: !88)
!88 = !DILexicalBlockFile(scope: !72, file: !89, discriminator: 0)
!89 = !DIFile(filename: "/root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ub_checks.rs", directory: "", checksumkind: CSK_MD5, checksum: "6ffce6e5ae865cd4e1a529f06e3329d2")
!90 = !DILocation(line: 75, column: 17, scope: !88)
!91 = !DILocation(line: 2069, column: 17, scope: !72)
!92 = !DILocalVariable(name: "self", arg: 1, scope: !93, file: !94, line: 1020, type: !73)
!93 = distinct !DISubprogram(name: "add", linkageName: "_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add17h1171b2ede0480fe3E", scope: !95, file: !94, line: 1020, type: !98, scopeLine: 1020, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, templateParams: !35, retainedNodes: !100)
!94 = !DIFile(filename: "/root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/i8*/mut_ptr.rs", directory: "", checksumkind: CSK_MD5, checksum: "03f9fff5488f048affd5530a4bafc084")
!95 = !DINamespace(name: "{impl#0}", scope: !96)
!96 = !DINamespace(name: "mut_ptr", scope: !97)
!97 = !DINamespace(name: "i8*", scope: !8)
!98 = !DISubroutineType(types: !99)
!99 = !{!73, !73, !22}
!100 = !{!92, !101}
!101 = !DILocalVariable(name: "count", arg: 2, scope: !93, file: !94, line: 1020, type: !22)
!102 = !DILocation(line: 1020, column: 29, scope: !93, inlinedAt: !103)
!103 = distinct !DILocation(line: 2070, column: 36, scope: !72)
!104 = !DILocation(line: 1020, column: 35, scope: !93, inlinedAt: !103)
!105 = !DILocation(line: 74, column: 35, scope: !106, inlinedAt: !103)
!106 = !DILexicalBlockFile(scope: !93, file: !89, discriminator: 0)
!107 = !DILocation(line: 1047, column: 35, scope: !93, inlinedAt: !103)
!108 = !DILocation(line: 75, column: 17, scope: !106, inlinedAt: !103)
!109 = !DILocation(line: 1054, column: 18, scope: !93, inlinedAt: !103)
!110 = !DILocation(line: 2070, column: 50, scope: !72)
!111 = !DILocation(line: 2070, column: 17, scope: !72)
!112 = !DILocation(line: 2068, column: 13, scope: !72)
!113 = !DILocation(line: 2073, column: 6, scope: !59)
!114 = distinct !DISubprogram(name: "iter", linkageName: "_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$4iter17hffcb055fac345fcdE", scope: !6, file: !5, line: 1049, type: !115, scopeLine: 1049, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, templateParams: !126, retainedNodes: !139)
!115 = !DISubroutineType(types: !116)
!116 = !{!117, !134}
!117 = !DICompositeType(tag: DW_TAG_structure_type, name: "Iter<u64>", scope: !13, file: !12, size: 128, align: 64, flags: DIFlagPublic, elements: !118, templateParams: !126, identifier: "3e1b1b4392c3abd0589d66838fb62b4c")
!118 = !{!119, !128, !129}
!119 = !DIDerivedType(tag: DW_TAG_member, name: "i8*", scope: !117, file: !12, baseType: !120, size: 64, align: 64, flags: DIFlagPrivate)
!120 = !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<u64>", scope: !121, file: !12, size: 64, align: 64, flags: DIFlagPublic, elements: !122, templateParams: !126, identifier: "6f68783b8a6dcaa6583144ea9f9d44d1")
!121 = !DINamespace(name: "non_null", scope: !97)
!122 = !{!123}
!123 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !120, file: !12, baseType: !124, size: 64, align: 64, flags: DIFlagPrivate)
!124 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u64", baseType: !125, size: 64, align: 64)
!125 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!126 = !{!127}
!127 = !DITemplateTypeParameter(name: "T", type: !125)
!128 = !DIDerivedType(tag: DW_TAG_member, name: "end_or_len", scope: !117, file: !12, baseType: !124, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!129 = !DIDerivedType(tag: DW_TAG_member, name: "_marker", scope: !117, file: !12, baseType: !130, align: 8, offset: 128, flags: DIFlagPrivate)
!130 = !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<&u64>", scope: !31, file: !12, align: 8, flags: DIFlagPublic, elements: !2, templateParams: !131, identifier: "1f9bb39810f5acdd1c0ed5565b3b1b22")
!131 = !{!132}
!132 = !DITemplateTypeParameter(name: "T", type: !133)
!133 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&u64", baseType: !125, size: 64, align: 64)
!134 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[u64]", file: !12, size: 128, align: 64, elements: !135, templateParams: !2, identifier: "e31591d84c9ef1b169c1a3318b8e9b52")
!135 = !{!136, !138}
!136 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !134, file: !12, baseType: !137, size: 64, align: 64)
!137 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !125, size: 64, align: 64)
!138 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !134, file: !12, baseType: !22, size: 64, align: 64, offset: 64)
!139 = !{!140}
!140 = !DILocalVariable(name: "self", arg: 1, scope: !114, file: !5, line: 1049, type: !134)
!141 = !DILocation(line: 1049, column: 17, scope: !114)
!142 = !DILocation(line: 1050, column: 9, scope: !114)
!143 = !DILocation(line: 1051, column: 6, scope: !114)
