d4(o0).
d5(o4).
d0(o2).
d1(o3).
d5(o2).
d2(o4).
d4(o5).
d1(o1).
d2(o1).

d5(X) :- d4(X).
d3(X) :- d4(X).
d3(X) :- d2(X), not d1(X).
d3(X) :- d1(X), not d0(X).
d2(X) :- d1(X), not d4(X).
d1(X) :- d0(X).