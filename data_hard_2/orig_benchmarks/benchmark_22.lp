d4(o17).
d7(o8).
d10(o8).
d4(o12).
d13(o6).
d9(o9).
d12(o16).
d6(o10).
d8(o3).
d7(o9).
d8(o12).
d5(o5).
d10(o12).
d4(o8).
d1(o16).

d2(X) :- d6(X), d10(X), not d12(X).
d16(X) :- d2(X), d17(X), not d12(X).
d11(X) :- d8(X), d4(X), not d5(X).
d4(X) :- d3(X), d6(X), not d11(X).
d7(X) :- not d5(X), d16(X), d11(X).
d12(X) :- d1(X), d4(X), not d13(X).
d13(X) :- d3(X), not d10(X), d12(X).
d0(X) :- not d16(X), d4(X), d12(X).
d17(X) :- d7(X), not d6(X), d10(X).
d1(X) :- d8(X), not d16(X), d4(X).
d3(X) :- d6(X), d10(X), not d11(X).
d9(X) :- d5(X), not d14(X), d13(X).
d11(X) :- d17(X), not d16(X), d12(X).
d14(X) :- not d3(X), d17(X), d10(X).
d2(X) :- d8(X), d14(X), not d17(X).