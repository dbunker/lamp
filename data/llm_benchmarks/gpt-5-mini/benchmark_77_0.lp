d2(o3).
d1(o4).
d1(o0).
d0(o0).
d4(o4).
d0(o3).

d5(X) :- d0(X).
d0(X) :- d4(X).
d4(X) :- not d2(X), d1(X).
d3(X) :- d2(X), d1(X).
d1(X) :- d2(X).