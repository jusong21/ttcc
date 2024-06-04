def uproot_writeable(events, include=["events", "run", "luminosityBlock"]):
    ev = {}
    include = np.array(include)
    no_filter = False
    if len(include) == 1 and include[0] == "*":
        no_filter = False
    for bname in events.keys():
        print('bname', bname)
        try: 
            events[bname].fields
        except:
            print('except')
            if not no_filter and bname not in include:
                print('not include ', bname)
                continue
            ev[bname] = ak.fill_none(
                ak.packed(ak.without_parameters(events[bname])), -99
            )
            continue
        b_nest = {}
        no_filter_nest = False
        if all(np.char.startswith(include, bname) == False):
            continue
        include_nest = [
            i[i.find(bname) + len(bname) + 1 :]
            for i in include
            if i.startswith(bname)
        ]

        if len(include_nest) == 1 and include_nest[0] == "*":
            no_filter_nest = True

        if not no_filter_nest:
            mask_wildcard = np.char.find(include_nest, "*") != -1
            include_nest = np.char.replace(include_nest, "*", "")
        for n in events[bname].fields:
            ## make selections to the filter case, keep cross-ref ("Idx")
            if (
                not no_filter_nest
                and all(np.char.find(n, include_nest) == -1)
                and "Idx" not in n
                and "Flavor" not in n
            ):
                continue
            if not _is_rootcompat(events[bname][n]) and ak.num(
                events[bname][n], axis=0
            ) != len(flatten(events[bname][n])):
                continue
            # skip IdxG
            if "IdxG" in n:
                continue
            b_nest[n] = ak.fill_none(
                ak.packed(ak.without_parameters(events[bname][n])), -99
            )
        ev[bname] = ak.zip(b_nest)
    return ev
