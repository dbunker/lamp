d3(o0).
d1(o3).
d0(o1).
d2(o3).
d1(o0).

d3(X) :- not d2(X), d0(X).
d1(X) :- not d0(X), d2(X).
d1(X) :- not d0(X), d2(X).