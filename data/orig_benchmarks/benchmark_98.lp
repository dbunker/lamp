d4(o3).
d3(o3).
d0(o1).
d2(o0).
d1(o1).

d4(X) :- not d0(X), d2(X).
d3(X) :- not d2(X), d4(X).