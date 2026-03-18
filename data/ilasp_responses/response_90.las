%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.193s
%% Conflict analysis                       : 0.609s
%%   - Positive Examples                   : 0.609s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.133s
%% Total                                   : 0.973s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d0(V1).
d1(V1) :- d2(V1); not d4(V1).
d0(V1) :- d3(V1); not d2(V1).
d3(V1) :- d1(V1); not d5(V1).

